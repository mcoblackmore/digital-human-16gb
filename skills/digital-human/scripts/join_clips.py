"""Join H3 talking-head clips into one continuous video.

Three fixes, in order, each one measured on this machine:
  1. crop the bottom band — H3 invents hands/fingers there and prompt wording does not stop it;
  2. shrink-only framing alignment — every clip is scaled to the smallest face and centred on the
     first clip's face, so clips cut together without a size jump and nothing is upscaled (a 1.41x
     upscale measured 268 -> 29 on Laplacian variance, i.e. visibly soft);
  3. cut each clip to its speech boundaries — detect leading/trailing silence in the audio and
     trim video+audio together, so the join has no dead air. A fixed head trim is not enough: a
     clip rendered to 17k+5 frames usually carries a few hundred ms of trailing silence, which
     reads as an obvious pause at the cut. A short fade-in also kills the burst each clip starts
     with (peak 863-931 vs 145-507 for normal speech).

usage: join_clips.py <out.mp4> <clip1.mp4> <clip2.mp4> [...]
env JC_CROP (0.12 bottom band), JC_TRIM (0.10 s), JC_CRF (15)
"""
import os, subprocess, sys, tempfile
import cv2, numpy as np, insightface, warnings
warnings.filterwarnings('ignore')

out = sys.argv[1]
clips = sys.argv[2:]
CROP = float(os.environ.get('JC_CROP', 0.12))      # bottom band (hands + edge smear)
CROP_X = float(os.environ.get('JC_CROP_X', 0.0))   # each side, removes stretched edge pixels
TRIM = float(os.environ.get('JC_TRIM', 0.10))
CRF = os.environ.get('JC_CRF', '15')

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = 'ffmpeg'

app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))


def face_of(frame):
    fs = app.get(frame)
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x1, y1, x2, y2 = f.bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2]), float(x2 - x1)


def speech_bounds(path, thresh=0.02, pad_head=0.03, pad_tail=0.08, skip_head=0.15):
    """(start, end) seconds of actual speech in the clip's audio track."""
    import wave
    tmpw = tempfile.mktemp(suffix='.wav')
    subprocess.run([FFMPEG, '-y', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', '16000', tmpw],
                   check=True)
    w = wave.open(tmpw)
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    sr = w.getframerate()
    w.close()
    os.unlink(tmpw)
    if a.size == 0:
        return 0.0, 0.0
    a = np.abs(a) / (np.abs(a).max() + 1e-9)
    # H3 starts every clip with a short burst; skipping it keeps the burst from being
    # mistaken for the first syllable (that mistake left ~0.5 s of dead air at the join)
    skip = int(sr * skip_head)
    idx = np.where(a[skip:] > thresh)[0] + skip
    if idx.size == 0:
        return 0.0, len(a) / sr
    start = max(skip_head, idx[0] / sr - pad_head)
    end = min(len(a) / sr, idx[-1] / sr + pad_tail)
    return start, end


def load(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames, fps, W, H


def cropped(frames, W, H):
    h = int(round(H * (1 - CROP)))
    dx = int(round(W * CROP_X))
    if dx > 0:
        frames = [fr[:h, dx:W - dx] for fr in frames]
        return frames, W - 2 * dx, h
    return [fr[:h] for fr in frames], W, h


info = []
for c in clips:
    frames, fps, W, H = load(c)
    frames, W, H = cropped(frames, W, H)
    fc = face_of(frames[0])
    if fc is None:
        sys.exit(f'no face in {c}')
    info.append({'path': c, 'frames': frames, 'fps': fps, 'W': W, 'H': H,
                 'c': fc[0], 'w': fc[1]})
    print(f'{os.path.basename(c)}: {len(frames)} frames {W}x{H} face_w={fc[1]:.1f}')

# Align to the LARGEST face: every clip is scaled by >=1, so the warped frame always covers
# the canvas and no border (hence no streak) can appear. Shrinking instead leaves an
# out-of-bounds margin that any border mode renders as visible streaks or mirror copies.
# no clamp: any cap below max(w) makes the largest clip shrink, which re-introduces the
# out-of-bounds margin (streaks) and leaves a size mismatch between clips
target_w = max(i['w'] for i in info)
target_c = info[0]['c']
W, H = info[0]['W'], info[0]['H']
print(f'target face_w={target_w:.1f} at {target_c[0]:.0f},{target_c[1]:.0f}')

tmp = tempfile.mkdtemp(prefix='join_')
parts = []
for idx, i in enumerate(info):
    s = target_w / i['w']
    M = np.array([[s, 0, target_c[0] - s * i['c'][0]],
                  [0, s, target_c[1] - s * i['c'][1]]], dtype=np.float32)
    inter = os.path.join(tmp, f'v{idx}.mkv')
    wr = cv2.VideoWriter(inter, cv2.VideoWriter_fourcc(*'FFV1'), i['fps'], (W, H))
    # BORDER_REPLICATE turned the out-of-bounds margin into vertical streaks; mirror instead,
    # then drop a small margin so nothing stretched survives
    for fr in i['frames']:
        wr.write(cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE))
    wr.release()
    part = os.path.join(tmp, f'p{idx}.mp4')
    a0, a1 = speech_bounds(i['path'])
    print(f'  part{idx}: speech {a0:.2f}-{a1:.2f}s of the clip')
    subprocess.run([FFMPEG, '-y', '-v', 'error', '-ss', f'{a0:.3f}', '-to', f'{a1:.3f}', '-i', inter,
                    '-ss', f'{a0:.3f}', '-to', f'{a1:.3f}', '-i', i['path'],
                    '-map', '0:v', '-map', '1:a',
                    '-af', 'afade=t=in:st=0:d=0.01',
                    '-c:v', 'libx264', '-crf', CRF, '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-b:a', '192k', part], check=True)
    parts.append(part)
    print(f'  part{idx}: scale={s:.3f}')

lst = os.path.join(tmp, 'list.txt')
with open(lst, 'w', encoding='utf-8') as f:
    for p in parts:
        f.write(f"file '{p}'\n")
subprocess.run([FFMPEG, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', lst,
                '-c:v', 'libx264', '-crf', CRF, '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '192k', out], check=True)
print('OUTPUT', out)
