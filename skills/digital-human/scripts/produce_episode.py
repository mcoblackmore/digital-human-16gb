"""Produce a whole episode: TTS each line with the saved voice profile, render each line
with H3 (length snapped to 17k+5 frames at 24 fps), then concatenate.

usage: produce_episode.py <script.json> <profile.pt> <workdir> [width=1056] [height=608]
env: QWEN_TTS_BASE, H3_LORA, COMFY_URL
"""
import json, os, subprocess, sys, time

script, profile, work = sys.argv[1], sys.argv[2], sys.argv[3]
W = sys.argv[4] if len(sys.argv) > 4 else '1056'
H = sys.argv[5] if len(sys.argv) > 5 else '608'
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
os.makedirs(work, exist_ok=True)
tts_dir = os.path.join(work, 'tts')
os.makedirs(tts_dir, exist_ok=True)
clips = []

t_all = time.time()
data = json.load(open(script, encoding='utf-8'))
lines = data['lines']
print(f'lines={len(lines)} res={W}x{H}', flush=True)

for ln in lines:
    lid = ln['id']
    t0 = time.time()
    # 1) voice
    subprocess.run([py, os.path.join(here, 'tts_lines.py'), script, profile, tts_dir, '1', str(lid)],
                   check=True)
    wav = os.path.join(tts_dir, f'L{int(lid):02d}_t1.wav')
    if not os.path.exists(wav):
        print(f'  line {lid}: no wav', flush=True)
        continue
    dur = float(subprocess.run([py, '-c', f'import soundfile as sf;print(sf.info(r"{wav}").duration)'],
                               capture_output=True, text=True).stdout.strip())
    frames = int(dur * 24) + 1
    # snap up to 17k+5, hard cap at 192 frames (8 s): longer clips crash ComfyUI at 1056x608
    frames = ((frames - 5 + 16) // 17) * 17 + 5
    frames = min(frames, 192)
    print(f'  line {lid}: {dur:.2f}s -> {frames} frames  (tts {time.time()-t0:.0f}s)', flush=True)
    # 2) video
    env = dict(os.environ, H3_LENGTH=str(frames), H3_STEPS='4')
    r = subprocess.run([py, os.path.join(here, 'h3_lipsync.py'), ln['image'], wav, work,
                        ln.get('prompt', 'A man speaks to the camera in a warm study with bookshelves. '
                                         'Camera: locked-off tripod, static frame, no pan, no zoom.'),
                        W, H], env=env, capture_output=True, text=True)
    last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''
    print('   ', last or r.stderr.strip()[-300:], flush=True)
    try:
        got = json.loads(last).get('files') or []
    except Exception:
        got = []
    if got:
        clips.append(got[0])
    else:
        mp4s = [os.path.join(work, f) for f in os.listdir(work) if f.startswith('h3_sync') and f.endswith('.mp4')]
        if mp4s:
            clips.append(max(mp4s, key=os.path.getmtime))

print(f'clips={len(clips)}', flush=True)
if clips:
    lst = os.path.join(work, 'concat.txt')
    with open(lst, 'w', encoding='utf-8') as f:
        for c in clips:
            f.write(f"file '{c.replace(os.sep, '/')}'\n")
    out = os.path.join(work, 'episode.mp4')
    subprocess.run([py, '-c',
                    'import imageio_ffmpeg,subprocess,sys;'
                    'exe=imageio_ffmpeg.get_ffmpeg_exe();'
                    f'subprocess.run([exe,"-y","-v","error","-f","concat","-safe","0","-i",r"{lst}",'
                    '"-c:v","libx264","-crf","17","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k",'
                    f'r"{out}"]);print("concat ok")'], check=False)
    print('OUTPUT', out, flush=True)
print(f'TOTAL {time.time()-t_all:.0f}s', flush=True)
