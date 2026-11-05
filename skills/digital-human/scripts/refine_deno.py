"""Deno's "8plus2" refine pass: draft latent -> 1.8x video latent upscale -> 2-step resample
with the refiner LoRA, audio latent masked so only the picture is redrawn.

usage: refine_deno.py <video.latent> <audio.latent> <portrait.png> <voice.wav> <out_dir> [W] [H]
env H3_UNET, H3_CLIP, H3_VAE, H3_AVAE, H3_REFINER, H3_TOTAL_STEPS=8, H3_SPLIT=6, H3_UP=1.8
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
REFINER = os.environ.get('H3_REFINER', 'h3\\minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors')
TOTAL = int(os.environ.get('H3_TOTAL_STEPS', 8))
SPLIT = int(os.environ.get('H3_SPLIT', 6))
PROMPT = os.environ.get('H3_PROMPT', 'A man speaks to the camera in a warm study with bookshelves.')
SEED = int(os.environ.get('H3_SEED', 5))

g = {
    "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
    "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
    "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
    "img": {"class_type": "LoadImage", "inputs": {"image": comfy.upload(portrait)}},
    "aud": {"class_type": "LoadAudio", "inputs": {"audio": comfy.upload(voice)}},
    "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["unet", 0], "lora_name": REFINER, "strength_model": 1.0}},
    "latv": {"class_type": "LoadLatent", "inputs": {"latent": comfy.upload(vid_lat)}},
    "lata": {"class_type": "LoadLatent", "inputs": {"latent": comfy.upload(aud_lat)}},
    "up": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
        "latent": ["latv", 0], "model_name": 'minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors',
        "mode": "target dimensions", "mode.width": W, "mode.height": H, "align": 32,
        "enable_temporal_chunking": True, "force_unload": True, "device": "cuda", "precision": "fp16"}},
    "mask": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 64, "height": 64}},
    "masked": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["lata", 0], "mask": ["mask", 0]}},
    "cat": {"class_type": "LTXVConcatAVLatent", "inputs": {
        "video_latent": ["up", 0], "audio_latent": ["masked", 0]}},
    "ref": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["clip", 0], "prompt": PROMPT, "width": W, "height": H,
        "length": 124, "ref_image_size": "match", "vae": ["vae", 0], "audio_vae": ["avae", 0],
        "ref_images.ref_image_0": ["img", 0], "ref_audios.ref_audio_0": ["aud", 0]}},
    "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["ref", 0]}},
    "sched": {"class_type": "BasicScheduler", "inputs": {
        "model": ["lora", 0], "scheduler": "simple", "steps": TOTAL, "denoise": 1.0}},
    "split": {"class_type": "SplitSigmas", "inputs": {"sigmas": ["sched", 0], "step": SPLIT}},
    "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
    "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    "guider": {"class_type": "BasicGuider", "inputs": {"model": ["lora", 0], "conditioning": ["ref", 0]}},
    "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
        "sigmas": ["split", 1], "latent_image": ["cat", 0]}},
    "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
    "deca": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
    "save": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["dec", 0], "audio": ["deca", 0], "frame_rate": 24, "loop_count": 0,
        "filename_prefix": "deno_refine", "format": "video/h264-mp4",
        "pingpong": False, "save_output": True}},
}

t0 = time.time()
print(f'deno refine {W}x{H} steps {TOTAL}->{TOTAL-SPLIT} (split {SPLIT})', flush=True)
e = comfy.run(g, poll=10)
files = [f for f in comfy.download(e, out) if f.endswith('.mp4')]
print(json.dumps({"status": e.get('status', {}).get('status_str'), "files": files,
                  "seconds": round(time.time() - t0), "error": comfy.error(e)}, ensure_ascii=False))
