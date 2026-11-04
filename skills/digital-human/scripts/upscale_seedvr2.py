"""Upscale a video with SeedVR2 (3B fp8) through the ComfyUI API.

Not part of the original skill: added for 16 GB VRAM cards, where Wan's own
896x512 render is the ceiling and a diffusion upscaler gets to 1280 faster.

usage: upscale_seedvr2.py <input.mp4> <out_dir> [target_width=1280] [seed=42]
env COMFY_URL, SV_DIT (seedvr2_ema_3b_fp8_e4m3fn.safetensors), SV_VAE (ema_vae_fp16.safetensors),
    SV_BATCH (5), SV_SWAP (0 blocks of the DiT kept in system RAM)
"""
import json, os, sys, time
import comfy

src, out = sys.argv[1], sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
DIT = os.environ.get('SV_DIT', 'seedvr2_ema_3b_fp8_e4m3fn.safetensors')
VAE = os.environ.get('SV_VAE', 'ema_vae_fp16.safetensors')
BATCH = int(os.environ.get('SV_BATCH', 5))
SWAP = int(os.environ.get('SV_SWAP', 0))

name = os.path.splitext(os.path.basename(src))[0]
g = {
    "vid": {"class_type": "VHS_LoadVideo", "inputs": {
        "video": comfy.upload(src), "force_rate": 0, "custom_width": 0, "custom_height": 0,
        "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1, "format": "None"}},
    "dit": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
        "model": DIT, "device": "cuda:0", "blocks_to_swap": SWAP, "swap_io_components": False,
        "offload_device": "cpu", "cache_model": True, "attention_mode": "sdpa"}},
    "vae": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
        "model": VAE, "device": "cuda:0", "encode_tiled": True, "encode_tile_size": 512,
        "encode_tile_overlap": 64, "decode_tiled": True, "decode_tile_size": 512,
        "decode_tile_overlap": 64, "offload_device": "cpu", "cache_model": True}},
    "up": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
        "image": ["vid", 0], "dit": ["dit", 0], "vae": ["vae", 0], "seed": seed,
        "resolution": target, "max_resolution": 0, "batch_size": BATCH, "uniform_batch_size": False,
        "color_correction": "lab", "offload_device": "cpu"}},
    "save": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["up", 0], "frame_rate": 25, "loop_count": 0,
        "filename_prefix": f"sv2_{name}", "format": "video/h264-mp4",
        "pingpong": False, "save_output": True, "audio": ["vid", 2]}},
}

t0 = time.time()
print(f'SeedVR2 upscale {target}px, batch {BATCH}, swap {SWAP}', flush=True)
e = comfy.run(g, poll=10)
files = [f for f in comfy.download(e, out) if f.endswith('.mp4')]
print(json.dumps({"status": e.get('status', {}).get('status_str'), "files": files,
                  "seconds": round(time.time() - t0), "error": comfy.error(e)}, ensure_ascii=False))
