"""Experiment 01: video-2 recipe (6-step Pika LoRA draft + one tiled refine step).

Runs one line of the episode through the new path and records timings plus QC numbers next to
the baseline, so the trade-off is a table rather than an impression.

usage: run_01.py <line id> [draft_w] [draft_h] [refine_w] [refine_h]
env: COMFY_URL, QWEN_TTS_BASE, H3_UNET, PIKA_LORA, H3_TILE_* (see refine_tiled.py)
"""
import json, os, subprocess, sys, time

line = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DW = sys.argv[2] if len(sys.argv) > 2 else '896'
DH = sys.argv[3] if len(sys.argv) > 3 else '512'
UW = sys.argv[4] if len(sys.argv) > 4 else '1600'
UH = sys.argv[5] if len(sys.argv) > 5 else '928'

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.environ.get('DH_SCRIPTS', r'E:\Digital_human_demo\skills\digital-human\scripts')
WORK = r'E:\Digital_human_demo\episode_hires'
LAB = r'E:\Digital_human_demo\lab\01_video2_recipe'
EPISODE = r'E:\Digital_human_demo\episode.json'
PROFILE = r'E:\Digital_human_demo\voice\lch.pt'
PORTRAIT = r'E:\Digital_human_demo\ref\locked_face_169.png'
PY = sys.executable
os.makedirs(LAB, exist_ok=True)
tag = f'L{line:02d}'
RUN = os.environ.get('RUN_TAG', 'a')   # separates experiments that reuse the same line
PROMPT = ('A man speaks directly to the camera in a warm study lined with bookshelves. '
          'He sits still with his hands out of frame, no gestures. '
          'Camera: locked-off tripod, static frame.')
PIKA = os.environ.get('PIKA_LORA', 'pika_h3_flex_ref2va_comfy_bf16.safetensors')
DRAFT_STEPS = os.environ.get('DRAFT_STEPS', '6')
SEED = os.environ.get('H3_SEED', '3')

result = {'run': os.environ.get('RUN_TAG', 'a'), 'line': tag, 'draft': f'{DW}x{DH}', 'refine': f'{UW}x{UH}',
          'draft_steps': DRAFT_STEPS, 'lora': PIKA}


RESTART = os.environ.get('COMFY_RESTART', r'E:\Digital_human_demo\start_comfy5.ps1')


def reset_comfy():
    """Free the GPU and the 32 GB RAM before a stage.

    Loading the 21 GB DenoMix while the 12.8 GB text encoder is still resident kills ComfyUI
    with a Windows access violation in torch.storage (measured, twice). A clean process per
    stage is cheaper than a crash.
    """
    subprocess.run(['powershell', '-NoProfile', '-Command',
                    "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*python_embeded*' } | "
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"],
                   capture_output=True, text=True)
    time.sleep(5)
    r = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', RESTART],
                       capture_output=True, text=True)
    ok = 'COMFY_UP' in (r.stdout or '')
    print(f'comfy reset: {"up" if ok else "FAILED"}', flush=True)
    return ok


def run(cmd, env=None, what=''):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    if r.returncode != 0:
        raise SystemExit(f'{what} failed rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}')
    return r, dt


def last_json(out):
    for ln in reversed(out.strip().splitlines()):
        try:
            return json.loads(ln)
        except Exception:
            continue
    return {}


# 1. voice
run([PY, os.path.join(SCRIPTS, 'tts_lines.py'), EPISODE, PROFILE, os.path.join(WORK, 'tts'),
     '1', str(line)], what='tts')
wav = os.path.join(WORK, 'tts', f'{tag}_t1.wav')
dur = float(subprocess.run([PY, '-c', f'import soundfile as sf;print(sf.info(r"{wav}").duration)'],
                           capture_output=True, text=True).stdout.strip())
frames = min(((int(dur * 24) + 1 - 5 + 16) // 17) * 17 + 5, 192)
result['speech_seconds'] = round(dur, 2)
result['frames'] = frames
print(f'{tag}: {dur:.2f}s -> {frames} frames', flush=True)

reset_comfy()

# 2. draft with the Pika adapter at 6 steps
env = dict(os.environ, H3_LENGTH=str(frames), H3_STEPS=DRAFT_STEPS, H3_SEED=SEED,
           H3_LORA=PIKA, H3_SAVE_LATENT=f'lab_{RUN}_{tag}')
r, dt = run([PY, os.path.join(SCRIPTS, 'h3_lipsync.py'), PORTRAIT, wav,
             os.path.join(LAB, 'draft'), PROMPT, DW, DH], env, 'draft')
draft = (last_json(r.stdout).get('files') or [None])[0]
if not draft:
    raise SystemExit(f'draft produced nothing:\n{r.stdout[-800:]}')
result['draft_seconds'] = round(dt)
print(f'draft ok {os.path.basename(draft)} {dt:.0f}s', flush=True)

q, _ = run([PY, os.path.join(SCRIPTS, 'qc_clips.py'), draft], what='qc draft')
result['qc_draft'] = q.stdout.strip()
print('qc draft:', q.stdout.strip(), flush=True)

reset_comfy()

# 3. one tiled refine step
OUTDIR = os.environ.get('COMFY_OUTPUT',
                        r'E:\ComfyUI_windows_portable_nvidia20260425\ComfyUI_windows_portable\ComfyUI\output')
import glob
vlat = max(glob.glob(os.path.join(OUTDIR, f'lab_{RUN}_{tag}_video_*.latent')), key=os.path.getmtime)
alat = max(glob.glob(os.path.join(OUTDIR, f'lab_{RUN}_{tag}_audio_*.latent')), key=os.path.getmtime)
r, dt = run([PY, os.path.join(SCRIPTS, 'refine_tiled.py'), vlat, alat, PORTRAIT, wav,
             os.path.join(LAB, 'refined'), UW, UH], what='tiled refine')
ref = (last_json(r.stdout).get('files') or [None])[0]
result['refine_seconds'] = round(dt)
print(f'refine ok {os.path.basename(ref) if ref else None} {dt:.0f}s', flush=True)

q, _ = run([PY, os.path.join(SCRIPTS, 'qc_clips.py'), ref], what='qc refine')
result['qc_refined'] = q.stdout.strip()
print('qc refined:', q.stdout.strip(), flush=True)

result['total_seconds'] = result['draft_seconds'] + result['refine_seconds']
json.dump(result, open(os.path.join(LAB, f'result_{RUN}.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(json.dumps(result, ensure_ascii=False, indent=1))
