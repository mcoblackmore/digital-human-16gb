"""Refine pass with the real H3 AV latent: load the saved draft latent, split picture/sound,
upscale only the picture latent, put the AV latent back together, then re-sample 3 steps with
the Open-H3-Regenerate-2K refiner LoRA (the video's "final touch").

usage: refine_av.py <latent.pt> <portrait.png> <voice.wav> <out_dir> [W=1920] [H=1088]
env H3_UNET, H3_CLIP, H3_VAE, H3_AVAE, H3_REFINER, H3_STEPS=3, H3_DENOISE=0.3
"""
import json, os, sys, time
import comfy

latent_file, portrait, voice, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
audio_latent_file = os.environ.get('H3_AUDIO_LATENT', '')
W = int(sys.argv[5]) if len(sys.argv) > 5 else 1920
H = int(sys.argv[6]) if len(sys.argv) > 6 else 1088
UNET = os.environ.get('H3_UNET', 'minimax_h3_ref2va_pruned_int8_convrot.safetensors')
CLIP = os.environ.get('H3_CLIP', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors')
VAE = os.environ.get('H3_VAE', 'minimax_h3_video_vae_fp16.safetensors')
AVAE = os.environ.get('H3_AVAE', 'minimax_h3_audio_vae_fp32.safetensors')
REFINER = os.environ.get('H3_REFINER', 'minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors')
STEPS = int(os.environ.get('H3_STEPS', 3))
DENOISE = float(os.environ.get('H3_DENOISE', 0.3))
PROMPT = os.environ.get('H3_PROMPT', 'A man speaks to the camera in a warm study with bookshelves.')

g = {
    "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
    "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
    "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
    "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AVAE}},
    "img": {"class_type": "LoadImage", "inputs": {"image": comfy.upload(portrait)}},
    "aud": {"class_type": "LoadAudio", "inputs": {"audio": comfy.upload(voice)}},
    "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": ["unet", 0], "lora_name": REFINER, "strength_model": 1.0}},
    "lat": {"class_type": "LoadLatent", "inputs": {"latent": comfy.upload(latent_file)}},
    "lata": {"class_type": "LoadLatent", "inputs": {"latent": comfy.upload(audio_latent_file)}},
    "up": {"class_type": "MinimaxH3LatentUpscaler3D", "inputs": {
        "latent": ["lat", 0], "model_name": 'minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors',
        "mode": "target dimensions", "mode.width": W, "mode.height": H, "align": 32,
        "enable_temporal_chunking": True, "force_unload": True, "device": "cuda", "precision": "fp16"}},
    "cat": {"class_type": "LTXVConcatAVLatent", "inputs": {
        "video_latent": ["up", 0], "audio_latent": ["lata", 0]}},
    "ref": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["clip", 0], "prompt": PROMPT, "width": W, "height": H,
        "length": 192, "ref_image_size": "match", "vae": ["vae", 0], "audio_vae": ["avae", 0],
        "ref_images.ref_image_0": ["img", 0], "ref_audios.ref_audio_0": ["aud", 0]}},
    "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["ref", 0]}},
    "sync": {"class_type": "PixaromaH3AudioSync", "inputs": {
        "model": ["lora", 0], "latent": ["cat", 0], "audio_vae": ["avae", 0], "track": ["aud", 0]}},
    "ks": {"class_type": "KSampler", "inputs": {
        "model": ["sync", 0], "positive": ["ref", 0], "negative": ["zero", 0],
        "latent_image": ["sync", 1], "seed": 42, "steps": STEPS, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": DENOISE}},
    "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
    "save": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["dec", 0], "audio": ["sync", 2], "frame_rate": 24, "loop_count": 0,
        "filename_prefix": "h3_refine_av", "format": "video/h264-mp4",
        "pingpong": False, "save_output": True}},
}

t0 = time.time()
print(f'refine AV {W}x{H} steps={STEPS} denoise={DENOISE}', flush=True)
e = comfy.run(g, poll=10)
files = [f for f in comfy.download(e, out) if f.endswith('.mp4')]
print(json.dumps({"status": e.get('status', {}).get('status_str'), "files": files,
                  "seconds": round(time.time() - t0), "error": comfy.error(e)}, ensure_ascii=False))
