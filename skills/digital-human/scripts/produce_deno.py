"""Produce a whole episode with the Deno 8plus2 draft settings, then join the clips.

usage: produce_deno.py <script.json> <profile.pt> <workdir> [width=896] [height=512]
env QWEN_TTS_BASE, COMFY_URL
"""
import json, os, subprocess, sys, time

script, profile, work = sys.argv[1], sys.argv[2], sys.argv[3]
W = sys.argv[4] if len(sys.argv) > 4 else '896'
H = sys.argv[5] if len(sys.argv) > 5 else '512'
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
os.makedirs(work, exist_ok=True)
tts_dir = os.path.join(work, 'tts')
os.makedirs(tts_dir, exist_ok=True)

lines = json.load(open(script, encoding='utf-8'))['lines']
print(f'lines={len(lines)} res={W}x{H}', flush=True)

t_all = time.time()
clips = []
for ln in lines:
    lid = ln['id']
    t0 = time.time()
    subprocess.run([py, os.path.join(here, 'tts_lines.py'), script, profile, tts_dir, '1', str(lid)],
                   check=True, capture_output=True)
    wav = os.path.join(tts_dir, f'L{int(lid):02d}_t1.wav')
    if not os.path.exists(wav):
        print(f'  line {lid}: no wav', flush=True)
        continue
    dur = float(subprocess.run([py, '-c', f'import soundfile as sf;print(sf.info(r"{wav}").duration)'],
                               capture_output=True, text=True).stdout.strip())
    frames = int(dur * 24) + 1
    frames = ((frames - 5 + 16) // 17) * 17 + 5     # H3 wants 17k+5
    frames = min(frames, 192)                        # hard cap: longer clips crash at this res
    print(f'  line {lid}: {dur:.2f}s -> {frames} frames (tts {time.time()-t0:.0f}s)', flush=True)
    env = dict(os.environ, H3_LENGTH=str(frames), H3_STEPS='8',
               H3_UNET='MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors',
               H3_LORA='h3\\MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors;'
                       'h3\\minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors')
    t1 = time.time()
    r = subprocess.run([py, os.path.join(here, 'h3_lipsync.py'), ln['image'], wav, work,
                        ln.get('prompt', 'A man speaks directly to the camera in a warm study '
                                         'lined with bookshelves. Camera: locked-off tripod, '
                                         'static frame.'),
                        W, H], env=env, capture_output=True, text=True)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-200:]
    print(f'    render {time.time()-t1:.0f}s: {tail}', flush=True)
    try:
        got = json.loads(tail).get('files') or []
    except Exception:
        got = []
    if got:
        clips.append(got[0])
    else:
        print('    NO OUTPUT, stopping', flush=True)
        break

print(f'clips={len(clips)}', flush=True)
if clips:
    out = os.path.join(work, 'episode_joined.mp4')
    r = subprocess.run([py, os.path.join(here, 'join_clips.py'), out] + clips,
                       capture_output=True, text=True)
    print(r.stdout.strip()[-500:], flush=True)
print(f'TOTAL {time.time()-t_all:.0f}s', flush=True)
