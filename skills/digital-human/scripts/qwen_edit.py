"""Run Qwen-Image-Edit-2511 (GGUF Q5_K_M + Lightning 4-step LoRA) through the ComfyUI API, one job per line.

The graph is ComfyUI's Qwen-Image-Edit 2511 template flattened to API form (same as workflows/形象编辑.json).

usage: qwen_edit.py <jobs.json> <out_dir>
jobs.json: [{"image": "start.png", "prompt": "...", "seed": 1, "name": "face_s1"}, ...]
  image   local file; or "blank:<W>x<H>" for a plain grey canvas (the whole picture comes from the prompt)
  image2  optional second reference image ("image 2" in the prompt): the locked face, a logo, a body
  turbo   default true: LoRA, 4 steps, cfg 1 (fast; the negative prompt does nothing)
          false: base model, 25 steps, cfg 4 (slower, finer texture, negative prompt works)
  steps, cfg, negative, denoise (< 1 refines instead of redrawing: structure and identity stay put)
Prints one JSON line per job with the downloaded files.
"""
import json, os, sys, time, struct, zlib, tempfile
import comfy

UNET = 'qwen-image-edit-2511-Q5_K_M.gguf'  # GGUF quantized for 16GB VRAM (was int8_convrot.safetensors)
CLIP = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
VAE = 'qwen_image_vae.safetensors'
LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'


def blank_png(spec):
    """Mid-grey PNG of the size in 'blank:WxH', written with the standard library."""
    w, h = map(int, spec.split(':', 1)[1].lower().split('x'))
    raw = b''.join(b'\x00' + b'\x80' * (3 * w) for _ in range(h))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    path = os.path.join(tempfile.gettempdir(), f'blank_{w}x{h}.png')
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
    return path


def graph(img, prompt, seed, prefix, neg='', turbo=True, steps=None, cfg=None, denoise=1.0, img2=None):
    steps = steps or (4 if turbo else 25)
    cfg = cfg if cfg is not None else (1.0 if turbo else 4.0)
    g = {
        "41": {"class_type": "LoadImage", "inputs": {"image": img}},
        "161": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": comfy.find_model('unet_gguf', UNET)}},
        "162": {"class_type": "CLIPLoader", "inputs": {"clip_name": comfy.find_model('text_encoders', CLIP),
                                                       "type": "qwen_image", "device": "default"}},
        "146": {"class_type": "VAELoader", "inputs": {"vae_name": comfy.find_model('vae', VAE)}},
        "160": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["41", 0]}},
        "151": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["162", 0], "vae": ["146", 0], "image1": ["160", 0], "prompt": prompt}},
        "149": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["162", 0], "vae": ["146", 0], "image1": ["160", 0], "prompt": neg}},
        "148": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["151", 0], "reference_latents_method": "index_timestep_zero"}},
        "147": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["149", 0], "reference_latents_method": "index_timestep_zero"}},
        "145": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["161", 0], "shift": 3.1}},
        "152": {"class_type": "CFGNorm", "inputs": {"model": ["145", 0], "strength": 1.0}},
        "153": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["152", 0], "lora_name": comfy.find_model('loras', LORA), "strength_model": 1.0}},
        "156": {"class_type": "VAEEncode", "inputs": {"pixels": ["160", 0], "vae": ["146", 0]}},
        "169": {"class_type": "KSampler", "inputs": {
            "model": ["153", 0] if turbo else ["152", 0], "positive": ["148", 0], "negative": ["147", 0],
            "latent_image": ["156", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "denoise": denoise}},
        "158": {"class_type": "VAEDecode", "inputs": {"samples": ["169", 0], "vae": ["146", 0]}},
        "195": {"class_type": "SaveImage", "inputs": {"images": ["158", 0], "filename_prefix": prefix}},
    }
    if not turbo:
        del g["153"]
    if img2:
        g["42"] = {"class_type": "LoadImage", "inputs": {"image": img2}}
        g["163"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["42", 0]}}
        g["151"]["inputs"]["image2"] = ["163", 0]
        g["149"]["inputs"]["image2"] = ["163", 0]
    return g


jobs, out = json.load(open(sys.argv[1], encoding='utf-8')), sys.argv[2]
uploaded = {}


def up(path):
    if path not in uploaded:
        uploaded[path] = comfy.upload(blank_png(path) if path.startswith('blank:') else path)
    return uploaded[path]


for j in jobs:
    t0 = time.time()
    e = comfy.run(graph(up(j['image']), j['prompt'], j.get('seed', 1), f"{comfy.SUB}/{j['name']}",
                        j.get('negative', ''), j.get('turbo', True), j.get('steps'), j.get('cfg'),
                        j.get('denoise', 1.0), up(j['image2']) if j.get('image2') else None))
    print(json.dumps({"name": j['name'], "status": e.get('status', {}).get('status_str'),
                      "files": comfy.download(e, out), "error": comfy.error(e),
                      "seconds": round(time.time() - t0)}, ensure_ascii=False), flush=True)
