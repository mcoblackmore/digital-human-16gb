---
name: digital-human
description: 用开源模型在本地做一个 AI 数字人主播：用文字写出脸（Qwen-Image-Edit-2511），用文字设计声音（Qwen3-TTS），再让它照着声音开口说话（InfiniteTalk + Wan2.1，跑在 ComfyUI 里）。每一步先量化检查，再拼对比图交给人挑。用户想做数字人、虚拟主播、AI 主持人，或者给已有的数字人换衣服、换背景、配新台词、做说话视频时使用。
---

# 数字人制作

这是 Builder² 做 AI 主播王渔鱼时实际走通的流程。你负责跑脚本、量结果、拼对比图；人负责说想要什么，再从对比图里挑。

脚本在 `scripts/`，ComfyUI 画布版的工作流在 `workflows/`（形象编辑、说话视频），模型清单和下载地址在 `reference/models.md`。

## 红线

- 脸和声音都从文字开始写。不拿任何真人的照片或录音当参考，名人、主播、朋友都不行。想参考谁的长相，只把特点写成文字：照片喂进去，模型会把那个人的脸一起搬过来。
- 发布时标明是 AI 生成：画面角上加标识，YouTube 选「更改或合成内容」，B 站勾「含 AI 生成内容」。
- 每一步出几个候选，拼成对比图交给人挑。人没挑之前，不往下走。

## 第 0 步：开工前检查

用户第一次提需求，先跑 `python scripts/check_setup.py`，把结果整理成一张清单回给用户。它会查 ComfyUI 连不连得上（默认 `http://127.0.0.1:8188`，别的电脑用环境变量 `COMFY_URL`），三个插件的节点在不在，11 个模型文件齐不齐，本机的 Python 包、CAM++ 和 ffmpeg。

缺什么，按 `reference/models.md` 告诉用户下哪个文件、放哪个文件夹。几个 G 的大文件，先说清大小，等用户同意再下。

## 第 1 步：脸

工具：`qwen_edit.py` 批量提交改图任务（Qwen-Image-Edit-2511，4 步提速 LoRA，每张约 35 秒），`face_check.py` 用 InsightFace 量脸，`contact_sheet.py` 拼对比图。

`qwen_edit.py` 的任务文件：

```json
[{"image": "blank:1248x832", "prompt": "...", "seed": 1, "name": "face_s1"},
 {"image": "start.png", "image2": "locked_face.png", "prompt": "...", "seed": 2, "name": "shirt_s2"}]
```

`image` 是要改的图，`blank:宽x高` 是一张灰底画布。`image2` 是第二张参考图，提示词里叫「图 2」。

1. **写五官。** 先跟用户问清楚年龄、性别、脸型、眉眼、发型、妆、表情、穿着和场景，写成一句一句的描述，比如：「更瘦的鹅蛋脸，下颌线清楚，眼神冷静专注，眉毛平直，淡妆，闭着嘴笑，二十八九岁」。起点可以是一张没有人的场景照片，也可以用灰底画布，提示词写「在画面中间加一位……」。同一句提示词换 4–6 个 seed。
2. **量和挑。** 每张跑 `face_check.py`，把数字标在对比图上（`contact_sheet.py out.jpg 3 512 a.png="sim 0.86" ...`），交给人挑。
3. **摆正头。** 编辑模型会照着原图的姿势画，提示词里不提，它不会顺手把头摆正。头歪超过 5 度，在下一次改图的提示词里写明「正对镜头，头摆正，不歪」，我们做王渔鱼时从 12 度降到了 3 度多。单独发一句「只把头摆正，其他不变」，我们试过几乎改不动。剩下 2–5 度，用 `level_head.py` 只转头部那一块。每一步做完都量一次。
4. **构图。** 跟它说「重新构图」，它只会把镜头往后拉，人越来越小。先用 `crop_face.py` 按脸的位置裁好，让脸占画面宽度两成左右，再把裁好的图整张重画一遍，提示词写清楚场景，要清晰的细节。
5. **加 logo 或道具。** 把 logo 图片当 `image2` 喂进去，提示词写「把图 2 的 logo 印在她的 T 恤胸前」。
6. **换头。** 身体那张图当 `image`，脸那张图当 `image2`，让它整张一次画出来。只换头、再回头修脖子，头会比原来小一圈，脖子怎么修都接不上。
7. **锁脸。** 人定下一张脸（下面叫定稿脸）以后，之后每次编辑都把定稿脸当 `image2` 一起喂进去，提示词里写「脸和图 2 保持一模一样」。出图后跑 `face_check.py --ref 定稿脸.png`，相似度低于 0.80 的换 seed 重来。日常换衣服、换背景，都走这一步。
8. **改成 16:9。** 不想动脸，就只补两边：`outpaint_169.py prep` 做出两边模糊的画布，让编辑模型把两边补成同一个场景，再 `outpaint_169.py merge` 把原图中间原样贴回去。提示词用英文，先说明两边只是模糊的占位，再写清左边、右边各接着画什么，比如：「This wide photo's left and right edges are only blurry placeholders. Extend the scene naturally into them: on the left, continue the bookshelf and the warm lamp glow; on the right, continue the bookshelf and the microphone boom arm running out of the frame. Keep the person and everything in the middle exactly the same.」只写「补全两边」的话，模型会原样留着那两条糊的。出 2–4 个 seed，按原尺寸看两边，挑接得最自然的。

## 第 2 步：声音

工具：`voice_design.py` 用文字造声音，`voice_knobs.py` 调三个旋钮，`voice_profile.py` 存成声音档案，`tts_lines.py` 配音，`tts_qc.py` 用 CAM++ 和 Whisper 质检。

1. **写声音。** 用文字描述一个人：年龄、性别、音色高低、语速、语气、场合。比如：「二十八九岁的华人女性，声音清澈，音色略低，知性沉稳又亲切，像在播客里和朋友聊天」。跑 `voice_design.py`，同一段描述出 4 个 seed，让人听了挑一个。
2. **调旋钮（人说「轻一点」「沉一点」时用）。** `voice_knobs.py` 改挑中的那段声音，三个旋钮是音高、共鸣和起伏。每个只动一点：音高、共鸣 3%–8%，起伏 10%–20%，出 3–4 档给人听。克隆的时候，模型会照着参考声音的音高和语调念，所以改参考就等于改声音。我们测过 13 位知识区女主播，音高从 163 到 254 赫兹，大多在 176–254 赫兹。男声我们没测过。
3. **混合（可选）。** 两个设计出来的声音可以按比例混：`voice_profile.py 参考.wav "参考念的原文" voice.pt --mix 另一个.wav --ratio 0.3`。
4. **存成档案。** 定下来的声音用 `voice_profile.py` 存成 `.pt`。声音设计模型每次出的人都不一样，之后每句都用这个档案让克隆模型念，声音才不会变。加 `--say "一句测试台词"` 先听三遍。
5. **配音。** 台词写成 `script.json`，格式见 `tts_lines.py` 开头。每句念 3–10 遍：`tts_lines.py script.json voice.pt tts/ 6`。
6. **质检。** `tts_qc.py script.json tts/ 参考.wav` 会自动挑，过关线见下面的检查标准。标了 `<-- check` 的句子，要给人听一下。
7. **多音字。** Whisper 听出来的字和稿子对不上，多半是多音字念错了，比如「角色」念成 jiǎo 色。在那句台词里加一个 `"tts"` 字段，换成同音字（角色→绝色，调 tiáo→条，长 zhǎng→涨），字幕仍然用 `"text"`。然后只重念这一句。

## 第 3 步：开口说话

两条路，按显存选：

- **16 GB 及以下显存 → 用 3A（MiniMax H3）**：小显存下 Wan 的 block swap 会把权重来回搬，GPU 空转，慢到不可用。
- **24 GB 以上显存 → 用 3B（Wan + InfiniteTalk）**：这是原始方案，底稿约束头动，最稳。

### 3A. MiniMax H3 ref2va + Pixaroma H3 Audio Sync（16 GB 显存实测可用）

工具：`h3_lipsync.py`（本仓库新增），插件 `ComfyUI-Pixaroma`，模型见 `reference/models.md`。

1. **一句话生成。** 不需要底稿视频：
   `H3_STEPS=4 H3_LORA=minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors H3_LENGTH=192 python h3_lipsync.py 定稿.png 配音.wav talk/ "英文提示词" 1056 608`
   `H3_LENGTH` 是帧数（24 fps，17 的倍数 + 5，192 = 8 秒）。
2. **音频必须走 `PixaromaH3AudioSync`。** 塞进 `MiniMaxH3ReferenceToVideo` 的 `ref_audios` 只会影响音色，模型会自己编语音和时序（实测嘴型与配音相关性 0.04）。AudioSync 才把音频锁进渲染。
3. **它说的是克隆音色，不是原录音。** 内容与台词一致、音色接近参考音频，但波形不同（实测相关 0.04、频谱重合 0.76）。要原录音逐字一致，只能走 3B。
4. **分辨率上限。** 16 GB 卡：864×480（241 秒/8 秒）、1056×608（471 秒）可跑；**1280×736 直接渲染会把 ComfyUI 打崩**（进程级 OOM），1056→1280 的 latent 放大也会崩。
5. **嘴动幅度。** 比 Wan 版小。提示词里写 `clear, pronounced lip movements: his mouth opens wide and closes fully on every syllable` 能加大一些。
6. **镜头会自己动。** H3 生成的画面里镜头会轻微漂移/呼吸。提示词写 `locked-off tripod, completely static frame, no pan, no zoom, no dolly` 能减弱到可接受的程度（实测中心抖动 11 px 左右，观感可接受）。
   **不要用整帧变换去"修"**：`face_lock.py` 试过两种模式（等比缩放锁定、只平移锁定），都会因为 H3 本身的镜头前后微动而被放大成明显的呼吸式推拉，比原片更假。实测结论：保持原始输出。
   真要完全静止，得用 24 GB 以上显存走 Wan + InfiniteTalk 的底稿路线（3B），底稿本身是静止的。
7. **要更高分辨率：走 Deno 的「8plus2」两段式**（见 3A-2），不要用 SeedVR2 或纯 latent 放大——前者嘴部飘，后者眼睛会糊。
8. **ComfyUI 启动参数**（缺一不可）：`--disable-cuda-malloc --disable-dynamic-vram --disable-nvml-pressure --disable-pinned-memory`。32 GB 内存还要把页面文件设到 32–64 GB，否则报 `os error 1455`。

### 3A-2. 高分辨率两段式（Deno「8plus2」，16 GB 实测 1600×928）

草稿用小尺寸快速出片，满意后再花 2 步把 latent 放大并重绘细节。**比直接渲染高分省一半显存，比后置超分更保脸。**

1. **草稿**（896×512 / 124 帧 / 8 步，双 LoRA），同时保存 AV latent：
   ```
   H3_UNET=MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors
   H3_STEPS=8
   H3_LORA=h3\MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors;h3\minimax_h3_dmd_ref2va_8step_turbo_pruned.safetensors
   H3_LENGTH=124
   H3_SAVE_LATENT=deno_L01
   python h3_lipsync.py 定稿.png 配音.wav draft/ "英文提示词" 896 512
   ```
   实测 **330 秒**。`H3_SAVE_LATENT` 会把 H3 的 AV latent 拆成 `*_video` / `*_audio` 两个 `.latent` 存到 ComfyUI/output（直接存整个 AV latent 会报 `'NestedTensor' object has no attribute 'contiguous'`）。
2. **精修**（1.8× latent + 2 步 + 音频屏蔽）：
   ```
   H3_UNET=MiniMax-H3_DenoMix_ref2va_pruned_int8_v3.safetensors
   H3_REFINER=minimax_h3_open_regenerate_2k_refiner_rank_16_bf16.safetensors
   python refine_deno.py <draft>_video_*.latent <draft>_audio_*.latent 定稿.png 配音.wav refined/ 1600 928
   ```
   实测 **531 秒**，峰值显存 12 GB。成片 1600×928，眼睛正常。
3. **三个必须对齐的细节**（少一个就失败）：
   - **精修只跑 2 步**：`BasicScheduler(simple, 8, 1)` → `SplitSigmas(step=6)` → 取**输出槽 1**（低噪声段）。取槽 0 是从头重画，出来是纯噪声。
   - **音频 latent 用 `SolidMask(0)` + `SetLatentNoiseMask` 屏蔽**，只让画面参与重采样；不屏蔽会把音频也重画，还会多占显存。
   - **草稿必须是 896×512 / 124 帧**。1056×608 / 192 帧的草稿做 3 步全量精修会直接把 ComfyUI 打崩（注意力开销约 2.4 倍）。
4. **更省时间的替代**：只做第 1 步（896×512 草稿 330 秒），分辨率够用就别精修。
5. **最终放大到 1080p+**：视频作者的免费方案是 RTX VSR（`DenoRTXVFXVideoFinisher`，不到 1 分钟）；付费方案约 6 分钟。都不如上面的 latent 精修保脸。

### 3B. Wan + InfiniteTalk（24 GB 以上显存）

工具：`still_source.py` 做底稿（不用模型），`infinitetalk_run.py` 提交 InfiniteTalk。

1. **底稿。** 把定稿图做成一段轻轻晃动、闭着嘴、没有声音的视频，长度是这句台词的时长加 0.5 秒：`still_source.py 定稿.png breathing_L01.mp4 6.8`
2. **生成。** `IT_PROMPT="用一句英文描述人物和场景" infinitetalk_run.py breathing_L01.mp4 tts/best/L01.wav talk/`。InfiniteTalk 一共画 4 步，先给底稿加一点噪声，再从第 2 步开始重画（`IT_START=2`），所以脸和动作都照着底稿，它只管把嘴加上去。默认 896×512，seed 2。在 DGX Spark 上，5 秒的一句话约 5 分钟（含加载模型），13 秒的一句话要 8–10 分钟。成片是 `talk_<台词名>_00001-audio.mp4`。
3. **不用底稿的情况。** 第一个参数写 `-`，再加 `IT_REF=定稿.png`，模型会直接拿照片生成。这样头动得多，大约是用底稿时的两倍，只在人想要更多动作时用。
4. **嘴和声音对齐。** 嘴会比声音早一点。把声音往后挪，中文 0.12 秒，英文 0.16 秒；画面最后一帧多停同样长，免得切掉最后一个字：

   ```
   ffmpeg -i talk.mp4 -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=0.12[v];[0:a]adelay=120|120[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 17 -pix_fmt yuv420p -c:a aac -b:a 192k L01.mp4
   ```

   电脑上没有 `ffmpeg` 命令的话，用 imageio-ffmpeg 自带的那个：`python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"` 会打出它的路径。

5. **检查。** 抽开头、中间、结尾三帧，跑 `face_check.py --ref 定稿.png`，低于 0.80 的换个 seed 重跑。嘴型给人看一遍。
6. **显存不够。** 我们在 Spark 上量到，896×512 生成时显存峰值约 24 GB。所以 24 G 显存的显卡要在 WanVideoWrapper 里给模型加载节点接一个 block swap 节点，或者换更小的压缩模型，会更慢。**16 GB 卡上即使 swap 30–40 层也会 OOM 或 GPU 空转**，直接走 3A。


## 实测选型（16 GB 显存，RTX 4060 Ti / 32 GB 内存）

同一句 7.4 秒的配音，两条路都跑通过，各有取舍：

| 路线 | 分辨率 | 耗时 | 嘴型 | 镜头 | 声音 |
|---|---|---|---|---|---|
| **Wan + InfiniteTalk**（3B） | 896×512 | 981 秒 | 跟配音 | 底稿约束，最稳 | **就是原录音** |
| **MiniMax H3**（3A） | 1056×608 | 460 秒 | 跟配音 | 会轻微漂移（提示词可减弱） | 克隆音色，非原录音 |

选择建议：

- **要 2 倍速度、要更高分辨率、能接受音色克隆** → 3A（H3，1056×608）
- **要原录音逐字一致、要镜头完全静止** → 3B（Wan，896×512）
- 两条都别超：Wan 上 896×512 是上限；H3 上 1056×608 是上限（1152×640 起必崩）

**整段长片不要指望一次跑完**：H3 单段超过 ~192 帧（8 秒）会崩 ComfyUI；Wan 的 198 帧也要开 VAE 分块。10 句台词 = 10 次渲染，16 GB 卡上合计 1 小时以上，先跟人确认值不值得。

## 检查标准

| 检查什么 | 用什么 | 过关线 | 说明 |
|---|---|---|---|
| 图里有几张脸 | `face_check.py` | 正好 1 张 | 多了说明编辑时贴进了第二个人 |
| 头歪不歪 | `face_check.py` 的 eye_line_deg | 2 度以内 | 2–5 度用 `level_head.py`，超过 5 度先在改图提示词里写明正对镜头 |
| 和定稿脸像不像 | `face_check.py --ref` | 0.80 以上 | 我们的经验线；同一张脸改衣服，一般在 0.85 上下 |
| 脸占画面多宽 | `face_check.py` 的 face_w | 近景 0.18–0.25 | 播客式近景 |
| 是不是同一个声音 | `tts_qc.py`，CAM++ | 0.6 以上 | 同一个声音 0.73–0.88，串成别人的只有 0.31–0.44。「嗨」开头的短句最容易串 |
| 念错字没有 | `tts_qc.py`，Whisper | 错字率 0.05 以下 | 超过就要听，多半是多音字 |
| 语速 | `tts_qc.py` | 中文每秒 5 字左右 | 挑最接近的那遍 |
| 嘴和声音 | ffmpeg | 声音后挪：中文 0.12 秒，英文 0.16 秒 | |

## 跟人怎么配合

- 每一步做完，拼一张带编号和检查数字的对比图，问人选哪个。声音直接给文件让人听。
- 人的话通常只有一句，比如「头歪了」「声音轻一点」「这个头动得太多了」。把它拆成具体任务：先量，再决定改哪一步，改完再量一遍。改前改后的数字都告诉人。
- 人说某张「挺自然」，就别在它上面再叠修复。
- 新的大模型、新的插件，先问人再下载、再安装。
