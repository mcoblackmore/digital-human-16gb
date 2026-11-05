"""Flag H3 clips that contain a smear / invented-hand artifact, so a pipeline can regenerate them.

H3 is stochastic: the same prompt and reference produce a clean clip on one seed and a smeared
one on the next. Human review catches it late and expensively, so score every clip right after
rendering and retry only the bad ones with a new seed.

Two cheap signals, both computed on the bottom band where H3 invents hands and stretches the torso:
  * sharpness dip — smeared/stretched pixels are blurry: Laplacian variance collapses
  * motion spike  — gestures and smears come with a sudden jump between frames

usage: qc_clips.py <clip.mp4> [...]      (prints one line per clip; exit code 1 if any is bad)
env QC_BAND (0.25 of the height, from the bottom), QC_ABS (150 absolute Laplacian variance —
clean clips measure 170-370, smeared ones 50-65), QC_SHARP (0.45 within-clip dip), QC_MOTION (12.0)
"""
import os, sys
import cv2
import numpy as np

BAND = float(os.environ.get('QC_BAND', 0.25))
SHARP = float(os.environ.get('QC_SHARP', 0.45))
ABS = float(os.environ.get('QC_ABS', 150.0))
MOTION = float(os.environ.get('QC_MOTION', 12.0))


def score(path):
    cap = cv2.VideoCapture(path)
    prev = None
    sharp, motion = [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        h = g.shape[0]
        band = g[int(h * (1 - BAND)):, :]
        sharp.append(cv2.Laplacian(band, cv2.CV_64F).var())
        if prev is not None:
            motion.append(float(np.abs(band.astype(np.int16) - prev.astype(np.int16)).mean()))
        prev = band
    cap.release()
    if not sharp:
        return None
    med = float(np.median(sharp))
    dip = min(sharp) / (med + 1e-9)          # 1.0 = uniform, low = some frames went blurry
    spike = max(motion) if motion else 0.0
    bad = med < ABS or dip < SHARP or spike > MOTION
    return med, dip, spike, bad


bad_any = False
for p in sys.argv[1:]:
    r = score(p)
    if r is None:
        print(f'BAD  {os.path.basename(p)}  unreadable')
        bad_any = True
        continue
    med, dip, spike, bad = r
    flag = 'BAD ' if bad else 'ok  '
    print(f'{flag} {os.path.basename(p)}  sharp_med={med:.1f} dip={dip:.2f} motion={spike:.1f}')
    bad_any = bad_any or bad
sys.exit(1 if bad_any else 0)
