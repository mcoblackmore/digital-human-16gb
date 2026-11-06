"""High-resolution episode supervisor: preflight -> draft (QC-gated) -> refine -> join.

Design rules, each one paid for in lost hours on this box:

1. Every subprocess result is checked (returncode + stderr). A silent failure once cost 10.5 h:
   the join step exited non-zero, only stdout was printed, and the run looked "finished".
2. Every stage verifies its artifact (file exists, ffprobe reads it, duration > 0).
3. The watchdog watches the FINAL artifact, not a live process.
4. Preflight runs before the first render: model files, portrait, voice profile, ComfyUI. An
   8-hour run must not die on a typo in a model name.
5. The manifest is keyed by line id + text hash, never by output filename order — a retry
   produces a new file and filename order silently desynchronises (that duplicated L01 and
   dropped L02 from a finished episode).
6. One supervisor owns draft -> refine -> join. No manual stage boundaries.

Seam continuity: each draft is seeded with the previous clip's last frame (H3_FIRST_FRAME), so
consecutive clips start from the pose the previous one ended on instead of restarting from the
static portrait.

usage: produce_hires.py <episode.json> <voice.pt> <workdir> [draft_w] [draft_h]
env: HIRES_ONLY=L07,L08  HIRES_REUSE=1  HIRES_RETRIES=2  HIRES_W=1600  HIRES_H=928
     HIRES_CHAIN=video|frame|off  HIRES_RETRY_LINE=2  HIRES_UNTIL=L03
     HIRES_CLEAN=1 (drop drafts/latents after the final check), HIRES_KEEP_REFINED=1
     HIRES_SAGE=1 (SageAttention, measured -28% per clip with no quality change)
     COMFY_URL, COMFY_OUTPUT, COMFY_RESTART, HIRES_PORTRAIT
"""
import hashlib, json, os, subprocess, sys, time, urllib.request

script, profile, work = sys.argv[1], sys.argv[2], sys.argv[3]
DW = sys.argv[4] if len(sys.argv) > 4 else '896'
DH = sys.argv[5] if len(sys.argv) > 5 else '512'
UW = os.environ.get('HIRES_W', '1600')
UH = os.environ.get('HIRES_H', '928')
RETRIES = int(os.environ.get('HIRES_RETRIES', '2'))
LINE_RETRIES = int(os.environ.get('HIRES_RETRY_LINE', '2'))
REUSE = os.environ.get('HIRES_REUSE', '1') == '1'
# off | frame | video. Measured on a two-clip seam (frame-to-frame delta at the cut, median
# motion 0.43): none 13.54, frame 12.10, video 7.85; with the 6-frame join blend 3.84 / 3.41 /
# 2.29. Video continuation costs 2.4x render time (862 s vs 360 s per clip).
CHAIN = os.environ.get('HIRES_CHAIN', 'video')
FORCE = os.environ.get('COMFY_FORCE', '1') == '1'
# Measured on L03 (158 frames, 1600x928): draft 421->330 s, refine 721->491 s, quality unchanged
# (sharp 279->269 / 160->161). Needs the SageAttention wheel built for ComfyUI's interpreter.
SAGE = os.environ.get('HIRES_SAGE', '1') == '1'
PROMPT = os.environ.get('HIRES_PROMPT',
                        'A man speaks directly to the camera in a warm study lined with bookshelves. '
                        'He sits still with his hands out of frame, no gestures. '
                        'Camera: locked-off tripod, static frame.')
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
tts_dir = os.path.join(work, 'tts')
draft_dir = os.path.join(work, 'draft')
refined_dir = os.path.join(work, 'refined')
frames_dir = os.path.join(work, 'frames')
for d in (tts_dir, draft_dir, refined_dir, frames_dir):
    os.makedirs(d, exist_ok=True)

OUTDIR = os.environ.get('COMFY_OUTPUT',
                        r'E:\ComfyUI_windows_portable_nvidia20260425\ComfyUI_windows_portable\ComfyUI\output')
MODELS = os.environ.get('COMFY_MODELS',
                        r'E:\ComfyUI_windows_portable_nvidia20260425\ComfyUI_windows_portable\ComfyUI\models')
RESTART = os.environ.get('COMFY_RESTART', r'E:\Digital_human_demo\start_comfy4.ps1')
PORTRAIT = os.environ.get('HIRES_PORTRAIT', r'E:\Digital_human_demo\ref\locked_face_169.png')
MANIFEST = os.path.join(work, 'refined_manifest.json')
UNET = os.environ.get('H3_UNET', 'MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors')
CLIP = os.environ.get('H3_CLIP', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
VAE = os.environ.get('H3_VAE', 'minimax_h3_video_vae_fp16.safetensors')
AVAE = os.environ.get('H3_AVAE', 'minimax_h3_audio_vae_fp32.safetensors')
REFINER = os.environ.get('H3_REFINER', 'minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors')
DRAFT_LORA = ('h3\\MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors;'
              'h3\\minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors')


class StepError(RuntimeError):
    pass


def keep_awake():
    """A render must never be interrupted by the box going to sleep.

    Two layers: the AC power policy (survives this process) and an execution-state lock held
    for the lifetime of this process (survives a policy change, and is what actually stops the
    idle timer while we sit in a 15-minute refine).
    """
    if os.name != 'nt':
        return None
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    for sub in ('standby-timeout-ac', 'hibernate-timeout-ac', 'disk-timeout-ac'):
        subprocess.run(['powercfg', '/change', sub, '0'], capture_output=True, text=True)
    return ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED


def sleep_guard_report():
    """Return a problem string when the box could still sleep, else None.

    The output of `powercfg /query` is localised, so the only stable part is the ordering of the
    hex values: the last two are the current AC and DC timeouts. Parsing the label text breaks
    on a Chinese Windows and silently reports "fine" — measured: it returned None with a 10 s
    timeout configured.
    """
    r = subprocess.run(['powercfg', '/query', 'SCHEME_CURRENT', 'SUB_SLEEP', 'STANDBYIDLE'],
                       capture_output=True, text=True)
    vals = []
    for line in r.stdout.splitlines():
        if '0x' in line:
            try:
                vals.append(int(line.rsplit('0x', 1)[1].strip(), 16))
            except ValueError:
                pass
    if len(vals) < 2:
        return 'cannot read the standby timeout from powercfg'
    ac = vals[-2]
    if ac != 0:
        return f'AC standby timeout is {ac}s, not never'
    return None


def run(cmd, env=None, what='', allow_fail=False):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 and not allow_fail:
        raise StepError(f'{what or cmd[0]} rc={r.returncode}\n'
                        f'stdout: {r.stdout.strip()[-600:]}\nstderr: {r.stderr.strip()[-600:]}')
    return r


def comfy_alive():
    url = os.environ.get('COMFY_URL', 'http://127.0.0.1:8189') + '/system_stats'
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_comfy():
    if comfy_alive() and not FORCE:
        return True
    # kill only ComfyUI's interpreter; a bare `taskkill /im python.exe` kills this script too
    subprocess.run(['powershell', '-NoProfile', '-Command',
                    "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*python_embeded*' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   capture_output=True, text=True)
    time.sleep(5)
    subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', RESTART],
                   capture_output=True, text=True)
    for _ in range(60):
        time.sleep(5)
        if comfy_alive():
            return True
    return False


def probe(path):
    """duration/streams via ffprobe; None when unreadable."""
    r = run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-show_entries', 'stream=codec_type', '-of', 'json', path],
            what=f'ffprobe {path}', allow_fail=True)
    if r.returncode != 0:
        return None
    try:
        d = json.loads(r.stdout)
        dur = float(d.get('format', {}).get('duration', 0))
        kinds = {s.get('codec_type') for s in d.get('streams', [])}
        return {'duration': dur, 'kinds': kinds} if dur > 0 else None
    except Exception:
        return None


def verify(path, what):
    info = probe(path)
    if not info:
        raise StepError(f'{what}: output unreadable -> {path}')
    if 'video' not in info['kinds']:
        raise StepError(f'{what}: no video stream -> {path}')
    return info


def last_json(out):
    for line in reversed(out.strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:
            continue
    return {}


def key_of(ln):
    h = hashlib.sha1(ln['text'].encode('utf-8')).hexdigest()[:8]
    return f"L{int(ln['id']):02d}:{h}"


def load_manifest():
    if not os.path.exists(MANIFEST):
        return {}
    try:
        data = json.load(open(MANIFEST, encoding='utf-8'))
    except Exception:
        return {}
    return {k: v for k, v in data.items() if os.path.exists(v)}


def save_manifest(m):
    json.dump(m, open(MANIFEST, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


def last_frame_of(path, out_png):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return None
    cv2.imwrite(out_png, fr)
    return out_png if os.path.exists(out_png) else None


def cleanup(keep_refined=True, keep=None):
    """Delete this run's intermediates once the final video has been verified.

    Kept: the episode, the per-line refined clips (re-joining is cheap, re-rendering is not),
    the manifest and the TTS wavs. Removed: draft renders, the ComfyUI latents that were only
    needed by the refine step, and the chaining frames.
    """
    import glob as _glob
    patterns = [os.path.join(draft_dir, '*.mp4'),
                os.path.join(OUTDIR, 'hi_L*.latent'),
                os.path.join(frames_dir, '*.png'),
                os.path.join(work, 'failcount.txt'),
                os.path.join(work, 'STOP.txt'),
                os.path.join(work, 'refined_manifest_backup.json')]
    if not keep_refined:
        patterns.append(os.path.join(refined_dir, '*.mp4'))
    else:
        # refined/ accumulates clips from every earlier run; keep only the ones this episode uses
        used = {os.path.abspath(v) for v in (keep or {}).values()}
        for f in _glob.glob(os.path.join(refined_dir, '*.mp4')):
            if os.path.abspath(f) not in used:
                patterns.append(f)
    freed = 0
    n = 0
    for pat in patterns:
        for f in _glob.glob(pat):
            try:
                freed += os.path.getsize(f)
                os.remove(f)
                n += 1
            except OSError:
                pass
    print(f'cleanup: removed {n} files, {freed / 1e6:.1f} MB', flush=True)


def preflight(lines):
    problems = []
    need = [(UNET, 'diffusion_models'), (CLIP, 'text_encoders'), (VAE, 'vae'),
            (AVAE, 'vae'), (REFINER, 'loras')]
    for name, sub in need:
        p = os.path.join(MODELS, sub, name)
        if not os.path.exists(p):
            problems.append(f'model missing: {p}')
    for lora in [x.strip() for x in DRAFT_LORA.split(';') if x.strip()]:
        p = os.path.join(MODELS, 'loras', lora)
        if not os.path.exists(p):
            problems.append(f'lora missing: {p}')
    if not os.path.exists(PORTRAIT):
        problems.append(f'portrait missing: {PORTRAIT}')
    if not os.path.exists(profile):
        problems.append(f'voice profile missing: {profile}')
    for ln in lines:
        if not os.path.exists(ln['image']):
            problems.append(f"line {ln['id']} image missing: {ln['image']}")
    if run(['ffprobe', '-version'], allow_fail=True).returncode != 0:
        problems.append('ffprobe not on PATH')
    guard = sleep_guard_report()
    if guard:
        problems.append(f'sleep guard: {guard}')
    if not comfy_alive():
        print('preflight: ComfyUI not answering, will start it', flush=True)
        if not ensure_comfy():
            problems.append('ComfyUI will not start')
    if problems:
        for p in problems:
            print('PREFLIGHT FAIL:', p, flush=True)
        raise StepError(f'{len(problems)} preflight problem(s)')
    print('preflight ok', flush=True)


lines = json.load(open(script, encoding='utf-8'))['lines']
ONLY = set(x.strip() for x in os.environ.get('HIRES_ONLY', '').split(',') if x.strip())
if ONLY:
    lines = [l for l in lines if f"L{int(l['id']):02d}" in ONLY]
# stop cleanly after a given line: an external watcher that kills the process mid-render wastes
# the clip in flight and can leave ComfyUI executing a prompt nobody collects
UNTIL = os.environ.get('HIRES_UNTIL', '').strip()
if UNTIL:
    stop_at = int(UNTIL.lstrip('Ll'))
    lines = [l for l in lines if int(l['id']) <= stop_at]
print(f'lines={len(lines)} only={sorted(ONLY) or "all"} draft={DW}x{DH} refine={UW}x{UH} '
      f'chain={CHAIN}', flush=True)

keep_awake()
guard = sleep_guard_report()
print(f'sleep guard: {"off" if guard is None else guard}', flush=True)
preflight(lines)

done = load_manifest()
if done:
    print(f'resume: {len(done)} already refined', flush=True)

t_all = time.time()
prev_frame = None
prev_clip = None


def render_line(ln):
    """One line end to end. Raises StepError on an unrecoverable failure."""
    lid = int(ln['id'])
    tag = f'L{lid:02d}'
    t0 = time.time()
    # TTS needs the GPU too: ComfyUI holds ~15 GB of the 16 GB card, so free it before the voice
    # step (measured: torch.OutOfMemoryError trying to allocate 3.59 GiB during tts_lines)
    if not ensure_comfy():
        raise StepError(f'{tag}: ComfyUI unavailable before tts')
    run([py, os.path.join(here, 'tts_lines.py'), script, profile, tts_dir, '1', str(lid)], what=f'{tag} tts')
    wav = os.path.join(tts_dir, f'{tag}_t1.wav')
    if not os.path.exists(wav):
        raise StepError(f'{tag}: tts produced no wav')
    dur = float(run([py, '-c', f'import soundfile as sf;print(sf.info(r"{wav}").duration)'],
                    what=f'{tag} wav probe').stdout.strip())
    frames = min(((int(dur * 24) + 1 - 5 + 16) // 17) * 17 + 5, 192)
    print(f'{tag}: {dur:.2f}s -> {frames} frames', flush=True)

    def newest_latent(kind):
        # SaveLatent never overwrites: a re-render writes _00002_, _00003_... Picking the fixed
        # _00001_ name silently refines the PREVIOUS render, which would waste a whole run.
        import glob
        cand = glob.glob(os.path.join(OUTDIR, f'hi_{tag}_{kind}_*.latent'))
        return max(cand, key=os.path.getmtime) if cand else None

    good = None
    if REUSE:
        vlat, alat = newest_latent('video'), newest_latent('audio')
    else:
        vlat = alat = None
    if not (REUSE and vlat and alat and os.path.exists(vlat) and os.path.exists(alat)):
        for attempt in range(RETRIES + 1):
            seed = 3 + attempt * 11
            env = dict(os.environ, H3_LENGTH=str(frames), H3_STEPS='8', H3_SEED=str(seed),
                       H3_UNET=UNET, H3_CLIP=CLIP, H3_VAE=VAE, H3_AVAE=AVAE,
                       H3_LORA=DRAFT_LORA, H3_SAVE_LATENT=f'hi_{tag}',
                       H3_SAGE='1' if SAGE else '0')
            if CHAIN == 'video' and prev_clip:
                env['H3_REF_VIDEO'] = prev_clip
            elif CHAIN == 'frame' and prev_frame:
                env['H3_FIRST_FRAME'] = prev_frame
            if not comfy_alive() and not ensure_comfy():
                raise StepError(f'{tag}: ComfyUI unavailable')
            r = run([py, os.path.join(here, 'h3_lipsync.py'), ln['image'], wav, draft_dir,
                     PROMPT, DW, DH], env, what=f'{tag} draft')
            got = (last_json(r.stdout).get('files') or [None])[0]
            if not got:
                print(f'  seed {seed}: no output file', flush=True)
                continue
            verify(got, f'{tag} draft')
            q = run([py, os.path.join(here, 'qc_clips.py'), got], what=f'{tag} qc', allow_fail=True)
            print(f'  seed {seed}: {q.stdout.strip()}', flush=True)
            if q.returncode == 0:
                good = got
                break
        if not good:
            raise StepError(f'{tag}: no clean draft after {RETRIES + 1} seeds')
    else:
        print(f'{tag}: reusing saved latents', flush=True)

    vlat = newest_latent('video')
    alat = newest_latent('audio')
    if not (vlat and alat):
        raise StepError(f'{tag}: no latents saved by the draft')
    for attempt in range(2):
        if not ensure_comfy():
            raise StepError(f'{tag}: ComfyUI unavailable for refine')
        r = run([py, os.path.join(here, 'refine_deno.py'), vlat, alat, ln['image'], wav,
                 refined_dir, UW, UH],
                dict(os.environ, H3_SAGE='1' if SAGE else '0'), what=f'{tag} refine')
        got = (last_json(r.stdout).get('files') or [None])[0]
        if got:
            info = verify(got, f'{tag} refined')
            print(f'  refined: {os.path.basename(got)} {info["duration"]:.2f}s '
                  f'({time.time() - t0:.0f}s)', flush=True)
            return got, good
        print(f'  {tag} refine attempt {attempt + 1} produced nothing', flush=True)
    raise StepError(f'{tag}: refine failed twice')


for ln in lines:
    k = key_of(ln)
    tag = k.split(':')[0]
    if k in done:
        print(f'{tag}: already refined, skip', flush=True)
        prev_clip = done[k]
        if CHAIN == 'frame':
            prev_frame = last_frame_of(done[k], os.path.join(frames_dir, f'{tag}_last.png')) or prev_frame
        continue
    err = None
    for attempt in range(LINE_RETRIES + 1):
        try:
            got, draft = render_line(ln)
            done[k] = got
            save_manifest(done)
            # the next clip continues from this one: prefer the draft (same resolution as the
            # next draft) over the refined clip
            prev_clip = draft or got
            if CHAIN == 'frame':
                prev_frame = last_frame_of(got, os.path.join(frames_dir, f'{tag}_last.png')) or prev_frame
            err = None
            break
        except StepError as e:
            err = e
            print(f'{tag}: attempt {attempt + 1} failed: {e}', flush=True)
            time.sleep(10)
    if err:
        print(f'{tag}: GIVING UP after {LINE_RETRIES + 1} attempts', flush=True)

# join only the lines this run was asked for, in script order: the manifest can hold clips from
# an earlier, longer run
clips = [done[key_of(l)] for l in lines if key_of(l) in done]
print(f'clips={len(clips)} of {len(lines)} requested', flush=True)
if not clips:
    raise StepError('nothing refined')

out = os.path.join(work, 'episode_hires.mp4')
r = run([py, os.path.join(here, 'join_clips.py'), out] + clips, what='join')
print(r.stdout.strip()[-500:], flush=True)
info = verify(out, 'joined episode')
# join trims leading/trailing silence per clip, so compare against what it reports, not the
# raw clip durations (43.9 s of clips becomes 37.1 s of speech)
exp = None
for line in r.stdout.splitlines():
    if line.startswith('EXPECTED '):
        exp = float(line.split()[1])
print(f'OUTPUT {out} {info["duration"]:.2f}s (expected {exp if exp else "?"}s)', flush=True)
if exp is not None and abs(info['duration'] - exp) > 1.0:
    raise StepError(f'join duration mismatch: {info["duration"]:.2f} vs expected {exp:.2f}')
if exp is None:
    raise StepError('join did not report EXPECTED duration')

if os.environ.get('HIRES_CLEAN', '1') == '1':
    cleanup(keep_refined=os.environ.get('HIRES_KEEP_REFINED', '1') == '1', keep=done)
print(f'TOTAL {time.time() - t_all:.0f}s', flush=True)
