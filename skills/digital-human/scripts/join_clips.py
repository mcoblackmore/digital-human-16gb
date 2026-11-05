"""Join H3 talking-head clips into one continuous video.

Three fixes, in order, each one measured on this machine:
  1. crop the bottom band — H3 invents hands/fingers there and prompt wording does not stop it;
  2. shrink-only framing alignment — every clip is scaled to the smallest face and centred on the
     first clip's face, so clips cut together without a size jump and nothing is upscaled (a 1.41x
     upscale measured 268 -> 29 on Laplacian variance, i.e. visibly soft);
  3. trim the first JC_TRIM seconds of audio+video and fade in — each H3 clip starts with a burst
     (peak 863-931 vs 145-507 for normal speech).

usage: join_clips.py <out.mp4> <clip1.mp4> <clip2.mp4> [...]
env JC_CROP (0.12 bottom band), JC_TRIM (0.10 s), JC_CRF (15)
"""
import os, subprocess, sys, tempfile
import cv2, numpy as np, insightface, warnings
warnings.filterwarnings('ignore')

out = sys.argv[1]
clips = sys.argv[2:]
CROP = float(os.environ.get('JC_CROP', 0.12))
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

target_w = min(i['w'] for i in info)
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
    for fr in i['frames']:
        wr.write(cv2.warpAffine(fr, M, (W, H), flags=cv2.INTER_AREA,
                                borderMode=cv2.BORDER_REPLICATE))
    wr.release()
    part = os.path.join(tmp, f'p{idx}.mp4')
    subprocess.run([FFMPEG, '-y', '-v', 'error', '-ss', str(TRIM), '-i', inter,
                    '-ss', str(TRIM), '-i', i['path'],
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
