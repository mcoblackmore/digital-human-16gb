"""Audio-driven talking video with MiniMax H3 ref2va + Pixaroma H3 Audio Sync.

The voice in the output is our recording: PixaromaH3AudioSync locks the audio into the
model/latent, so the mouth follows our words (plain ref_audios only styles the voice).

usage: h3_lipsync.py <portrait.png> <voice.wav> <out_dir> [prompt] [width=864] [height=480]
env COMFY_URL, H3_UNET, H3_CLIP, H3_VAE, H3_AVAE, H3_STEPS (8), H3_CFG (1), H3_LENGTH (192),
    H3_SEED, H3_LORA (optional 4-step lora name)
"""
import json, os, sys, time
import comfy

portrait, voice, out = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = sys.argv[4] if len(sys.argv) > 4 else (
    'A man speaks directly to the camera in a warm study lined with bookshelves, '
    'natural head motion, clear lip movement matching the audio, steady gaze.')
W = int(sys.argv[5]) if len(sys.argv) > 5 else 864
H = int(sys.argv[6]) if len(sys.argv) > 6 else 480
UNET = os.environ.get('H3_UNET', 'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
CLIP = os.environ.get('H3_CLIP', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
VAE = os.environ.get('H3_VAE', 'minimax_h3_video_vae_fp16.safetensors')
AVAE = os.environ.get('H3_AVAE', 'minimax_h3_audio_vae_fp32.safetensors')
STEPS = int(os.environ.get('H3_STEPS', 8))
CFG = float(os.environ.get('H3_CFG', 1.0))
LEN = int(os.environ.get('H3_LENGTH', 192))
SEED = int(os.environ.get('H3_SEED', 42))
LORA = os.environ.get('H3_LORA')
UP_W = int(os.environ.get('H3_UP_W', 0))
UP_H = int(os.environ.get('H3_UP_H', 0))
UP_MODEL = os.environ.get('H3_UP_MODEL', 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors')
REF_VIDEO = os.environ.get('H3_REF_VIDEO')

g = {
    "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
    "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
    "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
    "img": {"class_type": "LoadImage", "inputs": {"image": comfy.upload(portrait)}},
    "aud": {"class_type": "LoadAudio", "inputs": {"audio": comfy.upload(voice)}},
    "ref": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["clip", 0], "prompt": prompt, "width": W, "height": H, "length": LEN,
        "ref_image_size": "match", "vae": ["vae", 0], "audio_vae": ["avae", 0],
        "ref_images.ref_image_0": ["img", 0], "ref_audios.ref_audio_0": ["aud", 0]}},
    "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["ref", 0]}},
    "sync": {"class_type": "PixaromaH3AudioSync", "inputs": {
        "model": ["unet", 0], "latent": ["ref", 1], "audio_vae": ["avae", 0], "track": ["aud", 0]}},
    "ks": {"class_type": "KSampler", "inputs": {
        "model": ["sync", 0], "positive": ["ref", 0], "negative": ["zero", 0],
        "latent_image": ["sync", 1], "seed": SEED, "steps": STEPS, "cfg": CFG,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
    "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
    "save": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["dec", 0], "audio": ["sync", 2], "frame_rate": 24, "loop_count": 0,
        "filename_prefix": "h3_sync", "format": "video/h264-mp4",
        "pingpong": False, "save_output": True}},
}

if REF_VIDEO:
    g["refvid"] = {"class_type": "VHS_LoadVideo", "inputs": {
        "video": comfy.upload(REF_VIDEO), "force_rate": 24, "custom_width": 0, "custom_height": 0,
        "frame_load_cap": LEN, "skip_first_frames": 0, "select_every_nth": 1, "format": "None"}}
    g["ref"]["inputs"]["ref_videos.ref_video_0"] = ["refvid", 0]
    g["ref"]["inputs"]["ref_video_audios.ref_video_audio_0"] = ["aud", 0]

if UP_W and UP_H:
    g["up"] = {"class_type": "PixaromaH3LatentUpscale", "inputs": {
        "latent": ["ks", 0], "upscale_model": UP_MODEL, "width": UP_W, "height": UP_H}}
    g["dec"]["inputs"]["samples"] = ["up", 0]

if LORA:
    g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["unet", 0], "lora_name": LORA, "strength_model": 1.0}}
    g["sync"]["inputs"]["model"] = ["lora", 0]

t0 = time.time()
print(f'H3 ref2va+AudioSync {W}x{H}, {LEN} frames, {STEPS} steps cfg {CFG}, lora={LORA}', flush=True)
e = comfy.run(g, poll=10)
files = [f for f in comfy.download(e, out) if f.endswith('.mp4')]
print(json.dumps({"status": e.get('status', {}).get('status_str'), "files": files,
                  "seconds": round(time.time() - t0), "error": comfy.error(e)}, ensure_ascii=False))
