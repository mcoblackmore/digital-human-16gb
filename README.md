# AI 数字人 · 16 GB 显存复刻版

在 **RTX 4060 Ti 16 GB / 32 GB 内存** 的 Windows 机器上，把 [builder2ai/builder2-digital-human](https://github.com/builder2ai/builder2-digital-human) 的流程完整跑通，并把大显存方案换成小显存方案。

原 skill 在 DGX Spark（128 GB 统一内存）上验证；本仓库记录 16 GB 显存上的**实测替换、踩坑与最终配置**。

## 硬件实测环境

| 项 | 值 |
|---|---|
| GPU | RTX 4060 Ti 16 GB |
| 内存 | 32 GB（页面文件需 32–64 GB） |
| 系统 | Windows + ComfyUI 便携版（0.43.0） |
| Python | 3.13（ComfyUI 内置）/ 3.12（外部脚本） |

## 三条链路

### 1. 做脸：Qwen-Image-Edit-2511

| 原方案 | 本仓库 |
|---|---|
| int8 20.5 GB | **GGUF Q5_K_M 15.4 GB**（`UnetLoaderGGUF`） |
| — | 4 步 Lightning LoRA 不变 |

- 出图 **39–75 秒/张**，同句提示词换 4–6 个 seed 挑
- 锁脸：定稿脸当 `image2`，提示词写「脸和图 2 保持一模一样」，`face_check.py --ref` 相似度 ≥ 0.80（实测 0.889–0.919）
- 16:9：`outpaint_169.py prep/merge`，只补两侧、中间原样贴回

### 2. 声音：Qwen3-TTS 克隆

- 录音 20–40 秒 + 原文 → `voice_profile.py 录音.wav "原文" voice.pt` 存成声音档案
- 之后每句 `tts_lines.py` 用这个档案念，**8 秒/句**
- 校验：`tts_qc.py`（CAM++ 同人判定 ≥ 0.6，Whisper 错字率 ≤ 0.05）

### 3A. 说话（16 GB 路线）：MiniMax H3 ref2va + Pixaroma H3 Audio Sync

| 项 | 值 |
|---|---|
| 主模型 | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` 19.5 GB |
| 音频锁定 | **`PixaromaH3AudioSync` 节点**（塞进 `ref_audios` 只会影响音色，模型会自己编语音） |
| 加速 | `minimax_h3_ref2v_turbo_4step` LoRA，4 步 |
| 实测 | **896×512 / 124 帧（5.2 秒）= 330 秒** |
| 上限 | 草稿 896×512 / 124 帧最稳；1056×608 / 192 帧做精修必崩 |
| 声音 | 克隆音色（内容一致，非原录音波形） |

```bat
set COMFY_URL=http://127.0.0.1:8189
set H3_STEPS=4
set H3_LORA=minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
set H3_LENGTH=192
python scripts\h3_lipsync.py 定稿脸.png 配音.wav 输出目录 "英文提示词" 896 512
```

#### 高分辨率：Deno「8plus2」两段式（实测 1600×928）

草稿 896×512 → 1.8× latent 放大 → **只跑 2 步**重绘细节。比直接渲染高分省一半显存，比后置超分保脸。

```bat
:: 第 1 段：草稿（8 步 + 双 LoRA），同时存 AV latent
set H3_UNET=MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors
set H3_STEPS=8
set H3_LORA=h3\MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors;h3\minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors
set H3_LENGTH=124
set H3_SAVE_LATENT=clip01
python scripts\h3_lipsync.py 定稿脸.png 配音.wav draft\ "提示词" 896 512

:: 第 2 段：精修到 1600×928（2 步 + 音频 mask）
set H3_UNET=MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors
set H3_REFINER=minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors
python scripts\refine_deno.py <draft>_video_*.latent <draft>_audio_*.latent 定稿脸.png 配音.wav refined\ 1600 928
```

| 阶段 | 实测 | 峰值显存 |
|---|---|---|
| 草稿 896×512 / 124 帧 | 330 秒 | ~13 GB |
| 精修 1600×928 / 2 步 | 531 秒 | 12 GB |

**三个必须对齐的细节**（少一个就失败）：
1. 精修只跑 2 步：`BasicScheduler(8)` → `SplitSigmas(6)` → 取**输出槽 1**（取槽 0 是从头重画，出来纯噪声）
2. 音频 latent 用 `SolidMask(0)` + `SetLatentNoiseMask` 屏蔽，只重画画面
3. 草稿必须 896×512 / 124 帧；1056×608 / 192 帧做 3 步全量精修会把 ComfyUI 打崩（注意力开销 2.4 倍）

### 3B. 说话（24 GB+ 路线）：Wan + InfiniteTalk

| 项 | 值 |
|---|---|
| 主模型 | `wan2.1-i2v-14b-480p-Q3_K_M.gguf`（Q5 会 OOM）+ InfiniteTalk 5.1 GB |
| 文本编码器 | `umt5-xxl-enc-fp8_e4m3fn`，且 **`device: cpu`**（省 6.7 GB 显存） |
| block swap | `IT_SWAP=24`（40 层里搬 24 层到内存） |
| 实测 | **896×512 / 8 秒视频 = 981 秒** |
| 声音 | **就是原录音**，逐字一致 |

```bat
set IT_MODEL=wan2.1-i2v-14b-480p-Q3_K_M.gguf
set IT_W=896
set IT_H=512
set IT_SWAP=24
python scripts\infinitetalk_run.py - 配音.wav 输出目录
```

## 选型对照

| 路线 | 分辨率 | 5.2 秒耗时 | 眼睛 | 嘴型 | 声音 |
|---|---|---|---|---|---|
| **H3 草稿（Deno 参数）** | 896×512 | **330 秒** | ✅ | 跟配音 | 克隆音色 |
| **H3 草稿 + 8plus2 精修** | **1600×928** | **861 秒** | ✅ | 跟配音 | 克隆音色 |
| H3 原生（旧参数） | 1056×608 | 460 秒 | ✅ | 跟配音 | 克隆音色 |
| H3 + 纯 latent 放大 | 1920×1088 | 740 秒 | ❌ 眼睛糊 | 跟配音 | 克隆音色 |
| H3 + SeedVR2 1.2× | 1278×736 | 1110 秒 | ✅ | 嘴部轻飘 | 克隆音色 |
| Wan + InfiniteTalk | 896×512 | 981 秒（8 秒） | ✅ | 跟配音 | **原录音** |

## 长视频：分段生成 + 一键拼接

H3 单段上限约 8 秒（192 帧）。1 分钟片子 = 12 段，用 `join_clips.py` 一次拼好：

```bat
python scripts\join_clips.py 成片.mp4 段1.mp4 段2.mp4 ...
```

它依次做三件事（每一步都是实测出来的）：

1. **裁掉底部 12%** —— H3 会在胸像构图的下缘凭空生成手指，提示词压不住
2. **缩小对齐** —— 统一缩到最小的那张脸、居中到第一段，**只缩不放**（1.41× 放大把锐度从 268 压到 29）
3. **裁掉每段前 0.1 秒** + 音频 10 ms 淡入 —— 每段开头有爆音（峰值 863 vs 正常 145）

| 检查项 | 拼接前 | 拼接后 |
|---|---|---|
| 人脸尺寸跳变 | −29.1% | **+0.9%** |
| 人脸位移 | −37 px | **−1.1 px** |
| 段首爆音 | 863 | **34** |

## 目录结构

```
skills/digital-human/
├── SKILL.md                     # 完整流程（3A / 3B 双路线）
├── reference/models.md          # 模型清单、下载地址、小显存系统设置
├── scripts/
│   ├── check_setup.py           # 开工前自检（ComfyUI / 节点 / 模型 / Python 包）
│   ├── comfy.py                 # ComfyUI HTTP 客户端（上传、提交、下载）
│   ├── qwen_edit.py             # 做脸（API 图，GGUF + Lightning 4 步）
│   ├── face_check.py            # 人脸量化（张数、头歪、脸占比、相似度）
│   ├── contact_sheet.py         # 拼对比图
│   ├── crop_face.py / level_head.py / outpaint_169.py
│   ├── voice_design.py / voice_knobs.py / voice_profile.py / tts_lines.py / tts_qc.py
│   ├── still_source.py          # 底稿视频（轻晃闭嘴，不用模型）
│   ├── infinitetalk_run.py      # Wan + InfiniteTalk（3B 路线）
│   ├── h3_lipsync.py            # MiniMax H3 + AudioSync（3A 路线）
│   ├── refine_deno.py           # 8plus2 精修（1.8× latent + 2 步）
│   ├── join_clips.py            # 长视频拼接（裁底/缩小对齐/裁爆音）
│   ├── refine_av.py             # 精修（3 步全量版，会 OOM，留作参考）
│   ├── upscale_seedvr2.py       # SeedVR2 后置放大
│   └── produce_episode.py       # 整段编排（配音 → 逐句渲染 → 拼接）
└── workflows/                   # ComfyUI 画布版工作流
```

## 模型路径（ComfyUI/models/）

| 用途 | 路径 | 大小 |
|---|---|---|
| 做脸主模型 | `diffusion_models/gguf/qwen-image-edit-2511-Q5_K_M.gguf` | 15.4 GB |
| 做脸编码器 | `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | 8.7 GB |
| 做脸 LoRA | `loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | 0.8 GB |
| H3 主模型 | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GB |
| H3 编码器 | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GB |
| H3 VAE | `vae/minimax_h3_video_vae_fp16.safetensors` + `minimax_h3_audio_vae_fp32.safetensors` | 5.4 GB |
| H3 LoRA | `loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.9 GB |
| Wan 主模型 | `diffusion_models/gguf/wan2.1-i2v-14b-480p-Q3_K_M.gguf` | 8.6 GB |
| Wan 编码器 | `text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors` | 6.7 GB |
| 放大 | `SEEDVR2/seedvr2_ema_3b_fp8_e4m3fn.safetensors` | 3.4 GB |

完整清单与下载地址见 `skills/digital-human/reference/models.md`。

## 必须的启动参数

```bat
python main.py --windows-standalone-build --enable-manager --listen 0.0.0.0 --port 8189 ^
  --disable-cuda-malloc --disable-dynamic-vram --disable-nvml-pressure --disable-pinned-memory
```

- 前三个：新版 CUDA 分配器 / DynamicVRAM 会和 WanVideoWrapper、H3 的显存策略打架
- `--disable-pinned-memory`：常驻内存 21.5 GB → 6.9 GB（32 GB 内存必需）
- 页面文件设 32–64 GB，否则报 `os error 1455`

## 已知限制

1. **H3 单段 ≤192 帧（8 秒）**，超过会把 ComfyUI 打崩（进程级 OOM，不是异常）
2. **1152×640 及以上直接渲染必崩**；1280 只能靠后置放大
3. **镜头会轻微漂移**；整帧变换修复会放大成"呼吸式推拉"，实测更假，建议保持原片
4. **H3 生成的是克隆音色**，不是原录音；要逐字一致走 Wan 路线
5. **16 GB 卡上做长片很慢**：1 分钟成品约 1.1 小时（只草稿）或 2.9 小时（1600×928 精修）
6. **H3 的 latent 是嵌套的 AV latent**：`SaveLatent` 存不了，必须先 `LTXVSeparateAVLatent` 拆成画面/声音两份
7. **精修不能全量重采样**：3 步全量会把 ComfyUI 打崩；必须是 2 步（SplitSigmas 槽 1）+ 音频 mask

## 致谢与许可

- 原始流程：[builder2ai/builder2-digital-human](https://github.com/builder2ai/builder2-digital-human)（MIT）
- 本仓库在其基础上做了 16 GB 显存适配：GGUF 量化、CPU 文本编码器、block swap、H3 + AudioSync 路线、SeedVR2 放大、显存上限实测
- License: MIT（见 LICENSE）
