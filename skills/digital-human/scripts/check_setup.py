"""Check everything the skill needs before starting, and print one checklist.

ComfyUI: reachable at COMFY_URL, the custom nodes the two workflows use, every model file.
This Python: the packages for the face checks and the voice steps, CAM++ (CAMPP_ONNX), ffmpeg.
The face and voice tools may live in different Python environments; run this once in each.

usage: check_setup.py
"""
import os, shutil, importlib.util
import comfy

MODELS = [  # (ComfyUI model folder, file, what it is)
    ('unet_gguf', 'qwen-image-edit-2511-Q5_K_M.gguf', 'face: Qwen-Image-Edit-2511 GGUF Q5_K_M'),
    ('text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors', 'face: Qwen2.5-VL text encoder'),
    ('vae', 'qwen_image_vae.safetensors', 'face: VAE'),
    ('loras', 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors', 'face: 4-step LoRA'),
    ('unet_gguf', 'wan2.1-i2v-14b-480p-Q5_K_M.gguf', 'talk: Wan2.1 I2V 14B GGUF Q5_K_M'),
    ('diffusion_models', 'Wan2_1-InfiniTetalk-Single_fp16.safetensors', 'talk: InfiniteTalk'),
    ('text_encoders', 'umt5-xxl-enc-fp8_e4m3fn.safetensors', 'talk: umt5 text encoder fp8'),
    ('vae', 'Wan2_1_VAE_bf16.safetensors', 'talk: Wan VAE'),
    ('loras', 'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors', 'talk: 4-step LoRA'),
    ('clip_vision', 'clip_vision_h.safetensors', 'talk: CLIP vision'),
    ('wav2vec2', 'wav2vec2-chinese-base_fp16.safetensors', 'talk: chinese wav2vec2'),
    ('diffusion_models', 'minimax_h3_ref2va_pruned_int8_convrot.safetensors', 'talk16: MiniMax H3 ref2va'),
    ('text_encoders', 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'talk16: H3 text encoder'),
    ('vae', 'minimax_h3_video_vae_fp16.safetensors', 'talk16: H3 video VAE'),
    ('vae', 'minimax_h3_audio_vae_fp32.safetensors', 'talk16: H3 audio VAE'),
    ('loras', 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors', 'talk16: H3 4-step LoRA'),
    ('latent_upscale_models', 'minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors', 'talk16: H3 latent upscaler'),
    ('SEEDVR2', 'seedvr2_ema_3b_fp8_e4m3fn.safetensors', 'upscale: SeedVR2 3B fp8'),
    ('SEEDVR2', 'ema_vae_fp16.safetensors', 'upscale: SeedVR2 VAE'),
    ('diffusion_models', 'MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors', 'hi-res: DenoMix ref2va'),
    ('loras', 'MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors', 'hi-res: 8-step Acc LoRA'),
    ('loras', 'minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors', 'hi-res: 8-step dmd LoRA'),
    ('loras', 'minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors', 'hi-res: refiner LoRA'),
]
NODES = {  # node class -> where it comes from
    'TextEncodeQwenImageEditPlus': 'ComfyUI itself (update ComfyUI)',
    'FluxKontextMultiReferenceLatentMethod': 'ComfyUI itself (update ComfyUI)',
    'CFGNorm': 'ComfyUI itself (update ComfyUI)',
    'WanVideoSampler': 'ComfyUI-WanVideoWrapper',
    'MultiTalkModelLoader': 'ComfyUI-WanVideoWrapper',
    'Wav2VecModelLoader': 'ComfyUI-WanVideoWrapper',
    'ImageResizeKJv2': 'ComfyUI-KJNodes',
    'GetImageSizeAndCount': 'ComfyUI-KJNodes',
    'VHS_LoadVideo': 'ComfyUI-VideoHelperSuite',
    'VHS_VideoCombine': 'ComfyUI-VideoHelperSuite',
    'MiniMaxH3ReferenceToVideo': 'ComfyUI itself (update ComfyUI)',
    'PixaromaH3AudioSync': 'ComfyUI-Pixaroma',
    'PixaromaH3LatentUpscale': 'ComfyUI-Pixaroma',
    'SeedVR2VideoUpscaler': 'ComfyUI-SeedVR2_VideoUpscaler',
    'MinimaxH3LatentUpscaler3D': 'Comfyui_Minimax_h3_latent_Upscaler',
    'SplitSigmas': 'ComfyUI itself',
    'SetLatentNoiseMask': 'ComfyUI itself',
}
PACKAGES = {  # import name -> pip name, step
    'cv2': ('opencv-python', 'face checks, 底稿'), 'insightface': ('insightface', 'face checks'),
    'onnxruntime': ('onnxruntime', 'face checks, CAM++'), 'qwen_tts': ('qwen-tts', 'voice'),
    'soundfile': ('soundfile', 'voice'), 'torchaudio': ('torchaudio', 'voice QC'),
    'whisper': ('openai-whisper', 'voice QC'), 'pyworld': ('pyworld', 'voice knobs, voice QC'),
}
rows = []
ok = lambda good, what, note='': rows.append(('OK  ' if good else 'MISS') + f'  {what}' + (f'  -- {note}' if note else ''))

try:
    st = comfy.call('/system_stats')
    dev = st.get('devices', [{}])[0]
    ok(True, f"ComfyUI {st.get('system', {}).get('comfyui_version', '?')} at {comfy.URL}",
       f"{dev.get('name', '?')}, {dev.get('vram_total', 0) / 2**30:.0f} GB")
    info = comfy.call('/object_info', timeout=120)
    for n, src in NODES.items():
        ok(n in info, f'node {n}', '' if n in info else f'install {src}')
    for folder, f, what in MODELS:
        found = [x for x in comfy.models(folder) if x.replace('\\', '/').split('/')[-1] == f]
        ok(bool(found), f'{what}: {folder}/{found[0] if found else f}',
           '' if found else 'download, see reference/models.md')
except Exception as e:
    ok(False, f'ComfyUI at {comfy.URL}', f'not reachable ({e.__class__.__name__}); start it or set COMFY_URL')

for mod, (pipname, step) in PACKAGES.items():
    has = importlib.util.find_spec(mod) is not None
    ok(has, f'python {mod} ({step})', '' if has else f'pip install {pipname} (in the environment for {step})')
campp = os.path.expanduser(os.environ.get('CAMPP_ONNX', ''))
ok(bool(campp) and os.path.exists(campp), 'CAM++ campplus.onnx', '' if campp and os.path.exists(campp)
   else 'download it and set CAMPP_ONNX, see reference/models.md')
ff = shutil.which('ffmpeg') or importlib.util.find_spec('imageio_ffmpeg')
ok(bool(ff), 'ffmpeg', '' if ff else 'install ffmpeg or pip install imageio-ffmpeg')
print('\n'.join(rows))
print(f"\n{sum(r.startswith('MISS') for r in rows)} missing")
