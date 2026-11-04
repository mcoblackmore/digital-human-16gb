"""Make the portrait talk: InfiniteTalk video-to-video through the ComfyUI API (Kijai's WanVideoWrapper).

Flattened from the wrapper's example wanvideo_2_1_14B_V2V_InfiniteTalk_example_02.json (same as
workflows/说话视频.json). The 底稿 (still_source.py) is resized and VAE-encoded as the starting latents, its
first frame is the identity / clip-vision reference, the voice goes through chinese-wav2vec2, and
Wan2.1-I2V-14B + InfiniteTalk (lightx2v 4-step LoRA) redraw the clip in 81-frame windows with 9
frames of overlap. Sampling starts at step IT_START of 4 with noise added to the 底稿, so face and
motion follow the 底稿 and the model only adds the mouth.

A source of "-" switches to image-to-video: no 底稿, the model draws head motion and mouth from
IT_REF and the voice alone (all steps sampled). The head moves about twice as much.

usage: infinitetalk_run.py <breathing.mp4 | -> <voice.wav> <out_dir> [seed=2]
env IT_W/IT_H (896x512), IT_START (2), IT_STEPS (4), IT_AUDIO_SCALE (1.5),
    IT_PROMPT: one English sentence describing the person and the room,
    IT_REF: local portrait used as identity reference instead of the 底稿's first frame
Writes the finished mp4 (with the voice) into out_dir. Then delay the voice by the lip offset
(see SKILL.md).
"""
import json, os, sys, time
import comfy

src, voice, out = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2
W, H = int(os.environ.get('IT_W', 896)), int(os.environ.get('IT_H', 512))
START = int(os.environ.get('IT_START', 2))
STEPS = int(os.environ.get('IT_STEPS', 4))
ASCALE = float(os.environ.get('IT_AUDIO_SCALE', 1.5))
SWAP = int(os.environ.get('IT_SWAP', 30))
IT_MODEL = os.environ.get('IT_MODEL', 'wan2.1-i2v-14b-480p-Q5_K_M.gguf')
POS = os.environ.get('IT_PROMPT') or ('A person talks to the camera in a cozy studio, natural expressions, '
                                      'natural lip movements, clear mouth, steady gaze.')
NEG = ('bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall '
       'gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, '
       'poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three '
       'legs, many people in the background, walking backwards')
m = comfy.find_model
name = os.path.splitext(os.path.basename(voice))[0]

g = {
    "lora": {"class_type": "WanVideoLoraSelect", "inputs": {
        "lora": m('loras', 'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors'), "strength": 1.0,
        "low_mem_load": False, "merge_loras": False}},
    "mt": {"class_type": "MultiTalkModelLoader", "inputs": {
        "model": m('diffusion_models', 'Wan2_1-InfiniTetalk-Single_fp16.safetensors')}},
    "swap": {"class_type": "WanVideoBlockSwap", "inputs": {
        "blocks_to_swap": SWAP, "offload_img_emb": True, "offload_txt_emb": True,
        "use_non_blocking": True, "prefetch_blocks": 1}},
    "model": {"class_type": "WanVideoModelLoader", "inputs": {
        "model": m('unet_gguf', IT_MODEL), "base_precision": "bf16",
        "quantization": "disabled", "load_device": "main_device", "attention_mode": "sdpa",
        "lora": ["lora", 0], "multitalk_model": ["mt", 0], "block_swap_args": ["swap", 0]}},
    "vae": {"class_type": "WanVideoVAELoader", "inputs": {
        "model_name": m('vae', 'Wan2_1_VAE_bf16.safetensors'), "precision": "bf16"}},
    "clipv": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": m('clip_vision', 'clip_vision_h.safetensors')}},
    "text": {"class_type": "WanVideoTextEncodeCached", "inputs": {
        "model_name": m('text_encoders', 'umt5-xxl-enc-fp8_e4m3fn.safetensors'), "precision": "bf16",
        "positive_prompt": POS, "negative_prompt": NEG, "quantization": "fp8_e4m3fn", "use_disk_cache": True,
        "device": "cpu"}},
    "vid": {"class_type": "VHS_LoadVideo", "inputs": {
        "video": None, "force_rate": 0, "custom_width": 0, "custom_height": 0, "frame_load_cap": 0,
        "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan"}},
    "resize": {"class_type": "ImageResizeKJv2", "inputs": {
        "image": ["vid", 0], "width": W, "height": H, "upscale_method": "lanczos", "keep_proportion": "crop",
        "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 16, "device": "cpu"}},
    "enc": {"class_type": "WanVideoEncode", "inputs": {
        "vae": ["vae", 0], "image": ["resize", 0], "enable_vae_tiling": True, "tile_x": 256, "tile_y": 256, "tile_stride_x": 192, "tile_stride_y": 192,
        "tile_stride_x": 144, "tile_stride_y": 128, "noise_aug_strength": 0.0, "latent_strength": 1.0}},
    "first": {"class_type": "GetImageRangeFromBatch", "inputs": {"images": ["resize", 0], "start_index": 0, "num_frames": 1}},
    "size": {"class_type": "GetImageSizeAndCount", "inputs": {"image": ["first", 0]}},
    "cve": {"class_type": "WanVideoClipVisionEncode", "inputs": {
        "clip_vision": ["clipv", 0], "image_1": ["size", 0], "strength_1": 1.0, "strength_2": 1.0, "crop": "center",
        "combine_embeds": "average", "force_offload": True, "tiles": 0, "ratio": 0.5}},
    "i2v": {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {
        "vae": ["vae", 0], "width": ["size", 1], "height": ["size", 2], "frame_window_size": 81, "motion_frame": 9,
        "force_offload": False, "colormatch": "disabled", "start_image": ["size", 0], "tiled_vae": False,
        "clip_embeds": ["cve", 0], "mode": "infinitetalk", "output_path": ""}},
    "audio": {"class_type": "LoadAudio", "inputs": {"audio": None}},
    "w2v": {"class_type": "Wav2VecModelLoader", "inputs": {
        "model": m('wav2vec2', 'wav2vec2-chinese-base_fp16.safetensors'), "base_precision": "fp16",
        "load_device": "main_device"}},
    "emb": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {
        "wav2vec_model": ["w2v", 0], "audio_1": ["audio", 0], "normalize_loudness": True, "num_frames": 1000,
        "fps": 25.0, "audio_scale": ASCALE, "audio_cfg_scale": 1.0, "multi_audio_type": "para"}},
    "samp": {"class_type": "WanVideoSampler", "inputs": {
        "model": ["model", 0], "image_embeds": ["i2v", 0], "text_embeds": ["text", 0], "samples": ["enc", 0],
        "multitalk_embeds": ["emb", 0], "steps": STEPS, "cfg": 1.0, "shift": 11.0, "seed": seed, "force_offload": True,
        "scheduler": "dpm++_sde", "riflex_freq_index": 0, "denoise_strength": 1.0, "batched_cfg": False,
        "rope_function": "comfy", "start_step": START, "end_step": -1, "add_noise_to_samples": True}},
    "dec": {"class_type": "WanVideoDecode", "inputs": {
        "vae": ["vae", 0], "samples": ["samp", 0], "enable_vae_tiling": True, "tile_x": 256, "tile_y": 256, "tile_stride_x": 192, "tile_stride_y": 192,
        "tile_stride_x": 144, "tile_stride_y": 128, "normalization": "default"}},
    "out": {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["dec", 0], "audio": ["audio", 0], "frame_rate": 25.0, "loop_count": 0,
        "filename_prefix": f"{comfy.SUB}/talk_{name}", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17,
        "save_metadata": False, "trim_to_audio": True, "pingpong": False, "save_output": True}},
}

REF = os.environ.get('IT_REF')
if REF:
    g["refimg"] = {"class_type": "LoadImage", "inputs": {"image": comfy.upload(REF)}}
    g["refrs"] = {"class_type": "ImageResizeKJv2", "inputs": {
        "image": ["refimg", 0], "width": W, "height": H, "upscale_method": "lanczos", "keep_proportion": "crop",
        "pad_color": "0, 0, 0", "crop_position": "center", "divisible_by": 16, "device": "cpu"}}
    g["size"]["inputs"]["image"] = ["refrs", 0]
if src == '-':
    if not REF:
        sys.exit('image-to-video needs IT_REF')
    for k in ('vid', 'resize', 'enc', 'first'):
        del g[k]
    del g["samp"]["inputs"]["samples"]
    g["samp"]["inputs"].update(start_step=0, add_noise_to_samples=False)
    START = 0
else:
    g["vid"]["inputs"]["video"] = comfy.upload(src)
g["audio"]["inputs"]["audio"] = comfy.upload(voice)

t0 = time.time()
print(f'InfiniteTalk {"I2V" if src == "-" else "V2V"} {W}x{H}, steps {STEPS} from {START}, audio_scale {ASCALE}', flush=True)
e = comfy.run(g, poll=10)
files = [f for f in comfy.download(e, out) if f.endswith('.mp4')]
print(json.dumps({"status": e.get('status', {}).get('status_str'), "files": files,
                  "seconds": round(time.time() - t0), "error": comfy.error(e)}, ensure_ascii=False))
