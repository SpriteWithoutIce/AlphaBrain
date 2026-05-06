# WorldModelVLA README（AlphaBrain）

这个文档只针对 **WorldModelVLA 训练线**（不含 CosmosPolicy 线）。

对应代码入口：
- 框架：[AlphaBrain/model/framework/WorldModelVLA.py](/home/handoff/Desktop/AlphaBrain/AlphaBrain/model/framework/WorldModelVLA.py)
- 编码器统一接口：[AlphaBrain/model/modules/world_model/base/interface.py](/home/handoff/Desktop/AlphaBrain/AlphaBrain/model/modules/world_model/base/interface.py)
- 训练脚本：[scripts/run_world_model/train/run_world_model.sh](/home/handoff/Desktop/AlphaBrain/scripts/run_world_model/train/run_world_model.sh)

---

## 1. WorldModelVLA 总体架构

WorldModelVLA 的统一结构是：

1. `WorldModelEncoder`（Cosmos / WAN / V-JEPA）提取视觉 token  
2. `Text Encoder`（native 或 lightweight）提取文本 token  
3. `CrossAttentionFusion` 做视觉-文本融合  
4. `GR00T FlowMatching Action Head` 预测未来动作序列

在本仓库里，`WorldModelVLA` 训练默认是：
- 框架名：`QwenGR00T`（action head）
- world model 后端：`cos2 / cos25_4gpu / wan22 / vjepa`
- 数据：LeRobot-LIBERO（`lerobot_datasets`）

---

## 2. 四个后端模型的架构说明

## 2.1 Cosmos 2.0（`backend: cosmos2-diffusers`）

配置参考：[configs/models/config_cos2.yaml](/home/handoff/Desktop/AlphaBrain/configs/models/config_cos2.yaml)

核心实现：[AlphaBrain/model/modules/world_model/cosmos/encoder.py](/home/handoff/Desktop/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/encoder.py)

架构要点：
- Backbone：`CosmosTransformer3DModel`（Diffusers 版 DiT）
- 视觉路径：`image -> WanVAE -> latent -> DiT`
- DiT层数：28 blocks（代码常量）
- 默认动作特征层：`feature_layer_id=18`（中层特征）
- 文本条件：优先使用预计算 `T5` embedding（也支持 legacy Reason1 兜底）
- 训练可同时开启视频预测损失（`forward_with_video_loss` 路径）

这条线的直觉是：用 Cosmos 的时空扩散表征做视觉 backbone，再接 GR00T 动作头学控制。

---

## 2.2 Cosmos 2.5（`backend: cosmos2.5-diffusers`）

配置参考：[configs/models/config_cos25_4gpu.yaml](/home/handoff/Desktop/AlphaBrain/configs/models/config_cos25_4gpu.yaml)

核心实现同样在：[AlphaBrain/model/modules/world_model/cosmos/encoder.py](/home/handoff/Desktop/AlphaBrain/AlphaBrain/model/modules/world_model/cosmos/encoder.py)

与 Cosmos 2.0 的主要差异：
- 仍是 Diffusion Transformer 主体，但 checkpoint 换成 Predict2.5
- 文本条件优先使用 **Reason1 投影后的预计算 embedding**
- 代码里对 2.5 分支明确走 `reason1_projected_text_embeddings.pkl`

也就是说：视觉骨架相近，但 2.5 在文本条件路径上更依赖 Reason1 体系。

---

## 2.3 WAN 2.2（`backend: wan2.2`）

配置参考：[configs/models/config_wan22.yaml](/home/handoff/Desktop/AlphaBrain/configs/models/config_wan22.yaml)

核心实现：[AlphaBrain/model/modules/world_model/wan/encoder.py](/home/handoff/Desktop/AlphaBrain/AlphaBrain/model/modules/world_model/wan/encoder.py)

架构要点：
- 支持两种 WAN 变体：
  - `ti2v-5B`（默认，30 层，dim=3072）
  - `t2v-A14B`（40 层，dim=5120）
- 视觉路径：`image -> WAN VAE -> WAN DiT`
- 默认动作特征层：`feature_layer_id=14`
- 文本条件：
  - 优先预计算 `UMT5` embedding
  - 否则在线加载 UMT5-XXL 编码

WAN 线本质也是“扩散视频模型做视觉 backbone + GR00T 动作头”。

---

## 2.4 V-JEPA 2.1（`backend: vjepa2`）

配置参考：[configs/models/config_vjepa.yaml](/home/handoff/Desktop/AlphaBrain/configs/models/config_vjepa.yaml)

核心实现：[AlphaBrain/model/modules/world_model/vjepa/encoder.py](/home/handoff/Desktop/AlphaBrain/AlphaBrain/model/modules/world_model/vjepa/encoder.py)

架构要点：
- Backbone：V-JEPA ViT（本仓库默认常用 `vjepa2_1_vitG_384.pt`）
- 输出是 dense patch tokens（非 diffusion latent）
- 预处理是 ImageNet 标准化 + ViT patch/tubelet 结构
- 代码明确说明：**V-JEPA 分支默认不做 video reconstruction loss**（只提供表征特征）

这条线更偏“高质量视频表征 encoder”，不是生成式预测范式。

---

## 3. Checkpoint 下载（你需要从哪里下）

下面给的是“建议官方源 + 可直接执行命令”。

前置：安装 HF CLI 并登录（若模型需要许可）
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

### 3.1 Cosmos 2.0（必需）
```bash
huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World \
  --local-dir data/pretrained_models/Cosmos-Predict2-2B-Video2World
```

### 3.2 Cosmos 2.5（必需）
```bash
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --local-dir data/pretrained_models/Cosmos-Predict2.5-2B-diffusers
```

### 3.3 Cosmos Reason1（Cos2.5 文本条件需要）
```bash
huggingface-cli download nvidia/Cosmos-Reason1-7B \
  --local-dir data/pretrained_models/Cosmos-Reason1-7B
```

### 3.4 WAN 2.2（WAN 后端必需）
```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir data/pretrained_models/Wan2.2-TI2V-5B
```

### 3.5 V-JEPA 2.1 ViT-G（V-JEPA 后端必需）
```bash
mkdir -p data/pretrained_models/vjepa2
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt \
  -O data/pretrained_models/vjepa2/vjepa2_1_vitG_384.pt
```

### 3.6 T5-Small（Cos2 / V-JEPA 文本侧会用到）
```bash
huggingface-cli download google-t5/t5-small \
  --local-dir data/pretrained_models/t5-small
```

---

## 4. 与本仓库配置的路径对应关系

默认配置里通常是这些路径名（你下载后按这个目录组织最省事）：

- `data/pretrained_models/Cosmos-Predict2-2B-Video2World`
- `data/pretrained_models/Cosmos-Predict2.5-2B-diffusers`
- `data/pretrained_models/Cosmos-Reason1-7B`
- `data/pretrained_models/Wan2.2-TI2V-5B`
- `data/pretrained_models/vjepa2/vjepa2_1_vitG_384.pt`
- `data/pretrained_models/t5-small`

如果你的目录不一样，改对应 yaml：
- [configs/models/config_cos2.yaml](/home/handoff/Desktop/AlphaBrain/configs/models/config_cos2.yaml)
- [configs/models/config_cos25_4gpu.yaml](/home/handoff/Desktop/AlphaBrain/configs/models/config_cos25_4gpu.yaml)
- [configs/models/config_wan22.yaml](/home/handoff/Desktop/AlphaBrain/configs/models/config_wan22.yaml)
- [configs/models/config_vjepa.yaml](/home/handoff/Desktop/AlphaBrain/configs/models/config_vjepa.yaml)

---

## 5. 文本 embedding 预处理（强烈建议）

WorldModel 线建议先跑：

```bash
python scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_t5.py
python scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_reason1.py
python scripts/run_world_model/preprocess/precompute_text_embeddings/precompute_umt5.py
python scripts/run_world_model/preprocess/extract_nvidia_reason1_proj.py
```

对应文档：[scripts/run_world_model/README.md](/home/handoff/Desktop/AlphaBrain/scripts/run_world_model/README.md)

---

## 6. 训练命令（WorldModelVLA 线）

```bash
MODEL=cos2       bash scripts/run_world_model/train/run_world_model.sh
MODEL=cos25_4gpu bash scripts/run_world_model/train/run_world_model.sh
MODEL=wan22      bash scripts/run_world_model/train/run_world_model.sh
MODEL=vjepa      bash scripts/run_world_model/train/run_world_model.sh
```

---

## 7. 外部参考链接（下载与架构来源）

- Cosmos Predict2-2B-Video2World  
  https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World
- Cosmos Predict2.5-2B  
  https://huggingface.co/nvidia/Cosmos-Predict2.5-2B
- Cosmos Reason1-7B  
  https://huggingface.co/nvidia/Cosmos-Reason1-7B
- Wan2.2-TI2V-5B  
  https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- V-JEPA2 官方仓库（含 2.1 ckpt 表）  
  https://github.com/facebookresearch/vjepa2
- V-JEPA2.1 ViT-G 384 直链  
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt
- T5-Small  
  https://huggingface.co/google-t5/t5-small

