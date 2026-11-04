# 模型、插件和 Python 环境

下面是我们实际用的文件，文件名和 DGX Spark 上的一致。ComfyUI 的模型放进 `ComfyUI/models/` 下对应的文件夹。文件夹里再分子文件夹也行（比如 `diffusion_models/WanVideo/`），脚本会按文件名自己找。在画布里打开 `workflows/` 时，如果加载节点标红，在下拉框里重新选一下文件。

下载前先告诉用户文件多大，等用户同意再下。

## 做脸：约 31 GB

模型页：https://huggingface.co/Qwen/Qwen-Image-Edit-2511 （Apache 2.0）

| 文件夹 | 文件 | 大小 | 下载 |
|---|---|---|---|
| diffusion_models | qwen_image_edit_2511_int8_convrot.safetensors | 20.5 GB | https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models |
| text_encoders | qwen_2.5_vl_7b_fp8_scaled.safetensors | 9.4 GB | https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/text_encoders |
| vae | qwen_image_vae.safetensors | 0.25 GB | https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae |
| loras | Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors | 0.85 GB | https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning |

ComfyUI 左侧「模板」里有 Qwen-Image-Edit 2511 的官方模板，打开会列出同样的文件和下载链接。

## 开口说话（16 GB 显存路线）：MiniMax H3 ref2va，约 43 GB

小显存机器走这条（见 SKILL.md 3A）。模型页：https://huggingface.co/Comfy-Org/MiniMax-H3

| 文件夹 | 文件 | 大小 | 说明 |
|---|---|---|---|
| diffusion_models | minimax_h3_ref2va_pruned_int8_convrot.safetensors | 19.5 GB | ref2va = 参考图 + 音频 → 说话视频 |
| text_encoders | qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | 14.6 GB | H3 的文本/图像编码器（nvfp4 需要 CUDA 13） |
| vae | minimax_h3_video_vae_fp16.safetensors | 4.85 GB | 视频 VAE |
| vae | minimax_h3_audio_vae_fp32.safetensors | 0.58 GB | 音频 VAE |
| loras | minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors | 1.96 GB | 4 步 Turbo（Comfy-Org/MiniMax-H3 的 `loras/`） |
| latent_upscale_models | minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors | 0.69 GB | H3 专用 latent 放大（https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler） |
| SEEDVR2 | seedvr2_ema_3b_fp8_e4m3fn.safetensors + ema_vae_fp16.safetensors | 3.39 + 0.50 GB | 后置放大到 1280（https://huggingface.co/numz/SeedVR2_comfyUI） |

H3 的音频锁定靠 **ComfyUI-Pixaroma** 的 `PixaromaH3AudioSync` 节点，缺了它模型会自己编语音。

## 开口说话（24 GB 以上显存路线）：Wan + InfiniteTalk，约 36 GB

InfiniteTalk：https://github.com/MeiGen-AI/InfiniteTalk ，Wan2.1：https://github.com/Wan-Video/Wan2.1 （都是 Apache 2.0）

| 文件夹 | 文件 | 大小 | 下载 |
|---|---|---|---|
| diffusion_models | Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors | 17.0 GB | https://huggingface.co/Kijai/WanVideo_comfy |
| diffusion_models | Wan2_1-InfiniTetalk-Single_fp16.safetensors | 5.1 GB | 同上，`InfiniteTalk/` 文件夹 |
| text_encoders | umt5-xxl-enc-bf16.safetensors | 11.4 GB | 同上 |
| vae | Wan2_1_VAE_bf16.safetensors | 0.25 GB | 同上 |
| loras | lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors | 0.74 GB | 同上，`Lightx2v/` 文件夹 |
| clip_vision | clip_vision_h.safetensors | 1.3 GB | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/clip_vision |
| wav2vec2 | wav2vec2-chinese-base_fp16.safetensors | 0.19 GB | https://huggingface.co/Kijai/wav2vec2_safetensors （原版是腾讯的 https://huggingface.co/TencentGameMate/chinese-wav2vec2-base ） |

## ComfyUI 插件

在 ComfyUI 的插件管理器里搜名字安装，装完重启 ComfyUI。

| 插件 | 用途 | 我们用的版本 |
|---|---|---|
| ComfyUI-WanVideoWrapper https://github.com/kijai/ComfyUI-WanVideoWrapper | 在 ComfyUI 里跑 Wan2.1 和 InfiniteTalk | 088128b（2026-05-24） |
| **ComfyUI-Pixaroma** https://github.com/pixaroma/ComfyUI-Pixaroma | H3 的音频锁定（`PixaromaH3AudioSync`）、latent 放大、时间线 | main（2026-11） |
| **ComfyUI-SeedVR2_VideoUpscaler** https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler | 后置放大到 1280 | main（2026-11） |
| ComfyUI-KJNodes https://github.com/kijai/ComfyUI-KJNodes | 缩放图片、取尺寸 | f20bdd5（2026-10-19） |
| ComfyUI-VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite | 读视频、存视频 | 4d907be（2026-09-01） |
| ComfyUI-GGUF https://github.com/city96/ComfyUI-GGUF | 加载 GGUF 量化模型（16 GB 卡必需） | main |

## 小显存机器（16 GB）的启动与系统设置

这些不是模型，但缺了会直接失败：

1. **ComfyUI 启动参数**：`--disable-cuda-malloc --disable-dynamic-vram --disable-nvml-pressure --disable-pinned-memory`。前三个避免新版 CUDA 分配器/DynamicVRAM 与 WanVideoWrapper、H3 的显存策略打架；`--disable-pinned-memory` 把常驻内存从 21.5 GB 降到 6.9 GB（32 GB 内存机器必需）。
2. **Windows 页面文件**：设为 32–64 GB（我们放在 D 盘）。不够时报 `os error 1455`（提交上限耗尽）。
3. **跑之前关掉浏览器**：Chrome/Edge 能吃掉 3–6 GB。
4. **模型放 SSD**，E 盘至少要留 15 GB 余量。

ComfyUI 本体：856a922b（2026-09-10）。Qwen-Image-Edit 用到的节点是 ComfyUI 自带的，版本太旧会缺节点，更新 ComfyUI 即可。

## 声音：不在 ComfyUI 里

| 东西 | 装法 | 说明 |
|---|---|---|
| Qwen3-TTS https://github.com/QwenLM/Qwen3-TTS | `pip install qwen-tts`（我们用 0.1.1） | Apache 2.0。两个模型各约 4.3 GB，第一次运行自动下载：声音设计 Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign，克隆 Qwen/Qwen3-TTS-12Hz-1.7B-Base。已经下到本地的，用环境变量 `QWEN_TTS_DESIGN`、`QWEN_TTS_BASE` 指向文件夹。运行时提示找不到 SoX，可以不管，不影响生成。不装也能先在官方试用页上试：https://huggingface.co/spaces/Qwen/Qwen3-TTS |
| CAM++ 声纹模型 | 下载 `campplus.onnx`（28 MB）：https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | 设环境变量 `CAMPP_ONNX` 指向它。CAM++ 出自阿里的 3D-Speaker：https://github.com/modelscope/3D-Speaker |
| Whisper large-v3 https://github.com/openai/whisper | `pip install openai-whisper` | 第一次运行自动下载约 3 GB |
| pyworld | `pip install pyworld` | 调音高、共鸣、起伏 |

Mac 上跑 Qwen3-TTS 可以看社区项目 mlx-audio：https://github.com/Blaizzy/mlx-audio 。这个 skill 的脚本没在 Mac 上测过。

## 人脸检查

InsightFace：https://github.com/deepinsight/insightface ，`pip install insightface onnxruntime opencv-python`。第一次运行自动下载 buffalo_l 模型（约 330 MB）。

注意许可：InsightFace 的代码是 MIT，但它的预训练模型（包括 buffalo_l）官方只许非商业研究使用，商用要联系 InsightFace 拿授权，见它 README 的 License 一节。

## Python 环境

我们是分开装的，没测过全装进一个环境：

- 脸的检查和底稿：`opencv-python insightface onnxruntime numpy imageio-ffmpeg`
- 声音：`qwen-tts soundfile torchaudio openai-whisper pyworld onnxruntime`，需要英伟达显卡和 CUDA 版的 PyTorch
- `qwen_edit.py`、`infinitetalk_run.py`、`check_setup.py` 只用 Python 标准库，在哪台电脑上跑都行，用 `COMFY_URL` 指向 ComfyUI

## 硬件

我们只在 NVIDIA DGX Spark（GB10，128 GB 统一内存）上跑通过。用别的电脑主要看显存：

- 同一颗 GB10 芯片的机器（华硕 Ascent GX10、戴尔 Pro Max with GB10、联想 ThinkStation PGX、惠普 ZGX Nano）：同样 128 GB，应该可以照搬，没实测。
- 英伟达显卡台式机（RTX 4090 24 GB、RTX 5090 32 GB，系统内存 64 GB 以上）：开口说话要在 WanVideoWrapper 里开 block swap，或者换更小的压缩模型（fp8、GGUF）。没实测。
- Mac：声音能跑（mlx-audio），做脸能跑但慢，开口说话没找到跑通的可靠记录。
