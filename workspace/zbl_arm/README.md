# zbl 单臂后训练（Wall-OSS-0.5）

针对 zbl 6 轴单臂（自带眼在手相机）的微调配置与运行手册。

## 数据契约（zbl_arm_6a_datacol 录制）

| 特征 | 维度 | 语义 |
|---|---|---|
| `observation.state` | 10 | 末端位姿：位置3 + 6D旋转6 + 夹爪1（绝对值，基坐标系） |
| `action` | 10 | 同布局；`action[i] == state[i+1]`（下一帧到达位姿） |
| `observation.joints` | 7 | 关节 qpos（训练不消费，保留用于分析/审计） |
| `observation.images.wrist` | 640×480 | 眼在手单相机，h264 |

dof 声明：右臂 ee 键（真实 10 维）+ `action_padding: 16` = 26，匹配预训练动作空间。
loader 自动做 绝对→相对 转换（agent_pos 与 dof 同布局，转换良定义）。

## 一次性准备（本机）

```bash
mkdir -p /home/tangyang/workspace/zbl_arm_train
```

## 1. 计算归一化统计

```bash
cd /home/tangyang/workspace/wall-x-dev-ty_arm
/home/tangyang/miniconda3/envs/wallx/bin/python scripts/compute_norm_stats.py \
  --train_config workspace/zbl_arm/zbl_arm_ee.yml
```

产出 `/home/tangyang/workspace/zbl_arm_train/norm_stats.json`。**数据集增长后（如每采完
一批）需要重算**，训练 config 不用改。

## 2. 训练

正式（8×A100）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config workspace/zbl_arm/zbl_arm_ee.yml
```

有效 batch = 8卡 × 4 × 梯度累积2 = 64。A100 80G 显存富余时可把
`batch_size_per_gpu` 提到 8、`gradient_accumulation_steps` 降到 1。

冒烟（本机单卡）：

```bash
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config workspace/zbl_arm/zbl_arm_ee_smoke.yml
```

## 3. 推理与开环评测

```bash
bash scripts/run_serving.sh \
  --checkpoint-path /home/tangyang/workspace/zbl_arm_train/output/<ckpt> \
  --train-config-path workspace/zbl_arm/zbl_arm_ee.yml \
  --port 32195

# 对比模型预测与数据集真值
python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32195 \
  --dataset-root /home/tangyang/workspace/zbl_arm_6a_datacol/data/zbl_data/pick_up_test_tube_No._1_from_the_left_rack_and_place_it_in_the_right_rack \
  --train-config workspace/zbl_arm/zbl_arm_ee.yml \
  --episode-indices 0
```

单臂部署用 `--no-serialize-actions`（原始 32×26 chunk，前 10 维为
[pos3, rot6d, gripper]，后 16 维 padding 忽略），由单臂客户端切片下发。

## 迁移到 8×A100 机器的清单

1. **环境**（参照仓库根 README）：python 3.10 → `pip install -r requirements.txt`
   → flash-attn **必须为本机重编**：`FLASH_ATTN_CUDA_ARCHS="8.0" MAX_JOBS=8 pip
   install flash-attn==2.8.3 --no-build-isolation`（A100 是 sm_80；直接拷贝本机
   wheel 会报 no kernel image）→ dmuon → `pip install --no-deps -e lerobot`
   → `pip install --no-build-isolation -e .`
2. **数据**：拷贝数据集目录 `zbl_arm_6a_datacol/data/zbl_data/<task>/` 与
   `zbl_arm_train/norm_stats.json`。
3. **权重**：`checkpoints/wall-oss-0.5`（含 model.safetensors、config.json）与
   `checkpoints/Qwen2.5-VL-3B-Instruct`。
4. **改路径**：两份 yml 里的 `model.config_path`、`processor_path`、
   `pretrained_path`、`data.lerobot_config.repo_id`、`norm_stats_path`、
   `checkpoint.save_path`、`checkpoint.resume_from` 共 7 处绝对路径。

## 注意事项

- 夹爪行程参数必须全数据集一致（本批数据 gripper 上限 ≈1.0 rad），混用不同
  `--gripper-max` 的 episode 会引入分布裂缝。
- 数据按任务一目录一数据集组织；`repo_id` 指到**任务子目录**。多任务训练时逐任务
  配置或先合并数据集，并重算 norm_stats。
- 建议每个任务 50+ episode 后再启动正式训练；当前 1 条仅够管线冒烟。
