"""Video-2 recipe: draft latent -> latent upscale -> ONE tiled step, no refiner LoRA.

Deno's 2026-11-06 update drops the NVIDIA refiner LoRA and the 2-step second pass. The
expensive final pass instead runs as overlapping tiles, which is what lets a 16 GB card reach
1080p at 15 s. Everything else (DenoMix model, the same draft latent) stays identical, so this
script is a drop-in alternative to refine_deno.py and can be compared against it directly.

usage: refine_tiled.py <video.latent> <audio.latent> <portrait.png> <voice.wav> <out_dir> [W] [H]
env H3_UNET, H3_CLIP, H3_VAE, H3_AVAE, H3_TOTAL_STEPS=8, H3_SPLIT=7 (leaves 1 step),
    H3_TILE_W=896, H3_TILE_H=704, H3_OVL_X=256, H3_OVL_Y=320, H3_SEED
"""
import json, os, sys, time
import comfy

vid_lat, aud_lat, portrait, voice, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
W = int(sys.argv[6]) if len(sys.argv) > 6 else 1600
H = int(sys.argv[7]) if len(sys.argv) > 7 else 928
UNET = os.environ.get('H3_UNET', 'MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors')
CLIP = os.environ.get('H3_CLIP', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
VAE = os.environ.get('H3_VAE', 'minimax_h3_video_vae_fp16.safetensors')
AVAE = os.environ.get('H3_AVAE', 'minimax_h3_audio_vae_fp32.safetensors')
TOTAL = int(os.environ.get('H3_TOTAL_STEPS', 8))
SPLIT = int(os.environ.get('H3_SPLIT', 7))
REFINER = os.environ.get('H3_REFINER', '')   # empty = no LoRA (Deno's recipe); set to add detail back
TILE_W = int(os.environ.get('H3_TILE_W', 896))
TILE_H = int(os.environ.get('H3_TILE_H', 704))
OVL_X = int(os.environ.get('H3_OVL_X', 256))
OVL_Y = int(os.environ.get('H3_OVL_Y', 320))
PROMPT = os.environ.get('H3_PROMPT', 'A man speaks to the camera in a warm study with bookshelves.')
SEED = int(os.environ.get('H3_SEED', 5))

g = {
    "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
    "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
    "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
    "img": {"class_type": "LoadImage", "inputs": {"image": comfy.upload(portrait)}},
    "aud": {"class_type": "LoadAudio", "inputs": {"audio": comfy.upload(voice)}},
    "latv": {"class_type": "LoadLatent", "inputs": {"latent": comfy.upload(vid_lat)}},
    "lata": {"class_type": "LoadLatent", "inputs": {"latent": comfy.upload(aud_lat)}},
    "up": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
        "latent": ["latv", 0], "model_name": 'minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors',
        "mode": "target dimensions", "mode.width": W, "mode.height": H, "align": 32,
        "enable_temporal_chunking": True, "force_unload": True, "device": "cuda", "precision": "fp16"}},
    "cat": {"class_type": "LTXVConcatAVLatent", "inputs": {
        "video_latent": ["up", 0], "audio_latent": ["lata", 0]}},
    "ref": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["clip", 0], "prompt": PROMPT, "width": W, "height": H,
        "length": 124, "ref_image_size": "match", "vae": ["vae", 0], "audio_vae": ["avae", 0],
        "ref_images.ref_image_0": ["img", 0], "ref_audios.ref_audio_0": ["aud", 0]}},
    "sched": {"class_type": "BasicScheduler", "inputs": {
        "model": ["unet", 0], "scheduler": "simple", "steps": TOTAL, "denoise": 1.0}},
    "split": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["sched", 0], "step": SPLIT}},
    "tiled": {"class_type": "DenoMiniMaxH3TiledSampler", "inputs": {
        "model": ["unet", 0], "positive": ["ref", 0], "latent": ["cat", 0],
        "sigmas": ["split", 1], "seed": SEED,
        "tile_width": TILE_W, "tile_height": TILE_H,
        "overlap_x": OVL_X, "overlap_y": OVL_Y}},
    "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["tiled", 0], "vae": ["vae", 0]}},
    "deca": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["tiled", 0], "vae": ["avae", 0]}},
    "save": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["dec", 0], "audio": ["deca", 0], "frame_rate": 24, "loop_count": 0,
        "filename_prefix": "tiled_refine", "format": "video/h264-mp4",
        "pingpong": False, "save_output": True}},
}

if REFINER:
    # the tiled sampler alone produces a soft second pass on our upscale ratio (measured:
    # sharp 92 with 1 step, 99 with 2 steps, against 170 for the refiner LoRA path). The LoRA is
    # what restores detail; the tiles only buy VRAM headroom.
    g["lora"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["unet", 0], "lora_name": REFINER, "strength_model": 1.0}}
    g["tiled"]["inputs"]["model"] = ["lora", 0]
    g["sched"]["inputs"]["model"] = ["lora", 0]

t0 = time.time()
print(f'tiled refine {W}x{H} steps {TOTAL}->{TOTAL - SPLIT} tile {TILE_W}x{TILE_H} '
      f'overlap {OVL_X}x{OVL_Y} lora={REFINER or "none"}', flush=True)
e = comfy.run(g, poll=10)
files = [f for f in comfy.download(e, out) if f.endswith('.mp4')]
print(json.dumps({"status": e.get('status', {}).get('status_str'), "files": files,
                  "seconds": round(time.time() - t0), "error": comfy.error(e)}, ensure_ascii=False))
