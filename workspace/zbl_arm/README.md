# zbl 单臂后训练（Wall-OSS-0.5）

针对 zbl 6 轴单臂（眼在手单相机）的微调配置与集群训练手册。

**路径约定**：以下所有命令默认在**仓库根目录**执行（即 `wall-x/` 这一层）；仓库内文件一律用相对路径，集群上的外部资源（权重 / 数据 / 输出）统一用 `/path/to` 占位，提交前自行替换。

## 分工总览：采集机 / 集群开发机 / 网页平台

| 阶段 | 在哪 | 做什么 |
|---|---|---|
| 一次性 | **数据采集机**（遥操作那台） | 采集数据（zbl_arm_6a_datacol）→ 推送 `dev/ty_arm` 分支到 git 远端 → 上传数据集任务目录到集群存储 `/path/to/zbl_data/<task>` |
| 一次性 | **集群开发机**（login/dev 节点） | §1.1 拉代码 → §1.2 装环境（支持自定义镜像的话做成镜像，以后启任务免装）→ §1.3 放权重（集群可出网就直接从 HuggingFace 下载，否则从采集机上传）→ §2.1 算 norm_stats → §1.4 改 yml 的 7 处路径 → `wandb login` → §2.4 冒烟跑通 |
| 数据更新时 | **集群开发机** | 新一批数据上传后重算 norm_stats（§2.1，数据没变就跳过） |
| 每次训练 | **网页平台** | 选节点数/卡数 → 把 §2.2 的启动命令粘进"启动命令"框 → wandb 实时看曲线 |

环境与路径就绪后，日常只碰网页平台；只有数据更新才需要回集群开发机重算 norm_stats。

## 0. 数据契约（zbl_arm_6a_datacol 录制）

| 特征 | 维度 | 语义 |
|---|---|---|
| `observation.state` | 10 | 末端位姿：位置3 + 6D旋转6 + 夹爪1（绝对值，基坐标系） |
| `action` | 10 | 同布局；`action[i] == state[i+1]`（下一帧到达位姿） |
| `observation.images.wrist` | 640×480 | 眼在手单相机，h264 |

dof 声明：右臂 ee 键（真实 10 维）+ `action_padding: 16` = 26，匹配预训练动作空间。
loader 自动做 绝对→相对 转换（agent_pos 与 dof 同布局）。

---

## 1. 迁移步骤（集群）

### 1.1 取得代码

```bash
git clone <你的仓库地址> wall-x && cd wall-x
git checkout dev/ty_arm
```

（或直接 rsync 本地仓库目录；注意 `checkpoints/`、`lerobot/` 等是 gitignore 的本地资源，不含在内，需按 1.3 单独拷贝。）

### 1.2 环境（A100 机器，一次性）

```bash
conda create -n wallx python=3.10 -y && conda activate wallx
pip install -r requirements.txt

# flash-attn 必须在本机重编（A100 是 sm_80；拷贝别机的 wheel 会报 no kernel image）
FLASH_ATTN_CUDA_ARCHS="8.0" MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation

pip install "dmuon @ git+https://github.com/X-Square-Robot/dmuon.git"

# lerobot 用 --no-deps，避免覆盖 wall-x 依赖版本
git clone https://github.com/huggingface/lerobot.git /path/to/lerobot
pip install --no-deps -e /path/to/lerobot

# 安装 wall-x 本体（编译 CUDA 算子，需要几分钟）
MAX_JOBS=8 pip install --no-build-isolation -e .
```

### 1.3 拷贝外部资源

| 资源 | 来源 | 集群放置 | 说明 |
|---|---|---|---|
| 预训练权重 | HuggingFace `x-square-robot/wall-oss-0.5`（或从采集机上传 `checkpoints/wall-oss-0.5/`） | `/path/to/checkpoints/wall-oss-0.5` | 含 model.safetensors、config.json |
| Qwen 底座 | HuggingFace `Qwen/Qwen2.5-VL-3B-Instruct`（或从采集机上传） | `/path/to/checkpoints/Qwen2.5-VL-3B-Instruct` | tokenizer/processor 来源 |
| 数据集 | 数据采集机上传 `zbl_arm_6a_datacol/data/zbl_data/<task>/` | `/path/to/zbl_data/<task>` | 每任务一个子目录（完整 LeRobot v3 数据集） |
| norm_stats | **集群开发机上用 §2.1 生成** | `/path/to/zbl_arm_train/norm_stats.json` | 数据更新后重算 |

多机训练额外要求：**所有节点能看到相同路径**（NFS/Lustre 共享存储，或每节点放一份相同布局副本）；节点间 NCCL 网络可达；**节点必须同构（每节点卡数相同）**。

### 1.4 修改配置路径（`workspace/zbl_arm/zbl_arm_ee.yml` 共 7 处）

| 字段 | 改为 |
|---|---|
| `model.config_path` | `/path/to/checkpoints/wall-oss-0.5/config.json` |
| `model.processor_path` | `/path/to/checkpoints/Qwen2.5-VL-3B-Instruct` |
| `model.pretrained_path` | `/path/to/checkpoints/Qwen2.5-VL-3B-Instruct` |
| `data.lerobot_config.repo_id` | `/path/to/zbl_data/<task>`（指到任务子目录这一层） |
| `data.norm_stats_path` | `/path/to/zbl_arm_train/norm_stats.json` |
| `checkpoint.save_path` | `/path/to/zbl_arm_train/output` |
| `checkpoint.resume_from` | `/path/to/checkpoints/wall-oss-0.5/model.safetensors` |

### 1.5 wandb（实时监控）

配置文件里（`logging:` 段）：

```yaml
logging:
  use_wandb: true        # 已开启
  log_project: zbl_arm_finetune
  log_entity: ""         # 你的 wandb 用户名/团队名；个人默认空间可留空
  log_name: zbl_arm_ee   # run 名字
```

集群上认证（一次性，二选一）：

```bash
wandb login            # 交互式
export https_proxy="http://<USER>:<TOKEN>@proxy.pjlab.org.cn:23128/"
export http_proxy="$https_proxy"
export no_proxy="10.0.0.0/8,100.0.0.0/8,.pjlab.org.cn,127.0.0.1,localhost,::1"
export WANDB_API_KEY=xxx   # 非交互（推荐写进任务脚本）
```

计算节点无法出网时：`export WANDB_MODE=offline`（日志落本地 `wandb/` 目录，任务结束后 `wandb sync wandb/offline-run-*` 补传）。CLI 也可临时切：`--wandb_offline true/false`。多机时仅 rank 0 上报。

---

## 2. 训练步骤

### 2.1 归一化统计（在集群开发机执行；数据更新后重算，config 不用动）

```bash
python scripts/compute_norm_stats.py \
  --train_config workspace/zbl_arm/zbl_arm_ee.yml \
  --data_root /path/to/zbl_data/<task> \
  --output_path /path/to/zbl_arm_train/norm_stats.json
```

### 2.2 启动命令（网页平台"启动命令"框里填这个）

torchrun 用绝对路径调用，平台 shell 无需 `conda activate`（交互终端里 `conda activate wallx` 后可直接用 `torchrun`）。

**单机 8 卡任务：**

```bash
cd /root/workspace/wall-x && \
export https_proxy="http://<USER>:<TOKEN>@proxy.pjlab.org.cn:23128/" \
       http_proxy="$https_proxy" \
       no_proxy="10.0.0.0/8,100.0.0.0/8,.pjlab.org.cn,127.0.0.1,localhost,::1" && \
WANDB_API_KEY=<你的key> \
/root/workspace/miniconda3/envs/wallx/bin/torchrun --nproc_per_node=8 \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config workspace/zbl_arm/zbl_arm_ee.yml \
  --log_to_file
```

（平台已自动注入 `CUDA_VISIBLE_DEVICES` 时不用手动设；没注入就前置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`。）

**多机多卡任务**（以 2 机 × 8 卡为例），按平台注入方式二选一：

方式 A——平台注入标准环境变量（`MASTER_ADDR` / `MASTER_PORT` / `NODE_RANK` / `NNODES`，多数平台是这个模式，变量名以平台文档为准）：

```bash
cd /root/workspace/wall-x && \
export https_proxy="http://<USER>:<TOKEN>@proxy.pjlab.org.cn:23128/" \
       http_proxy="$https_proxy" \
       no_proxy="10.0.0.0/8,100.0.0.0/8,.pjlab.org.cn,127.0.0.1,localhost,::1" && \
WANDB_API_KEY=<你的key> \
/root/workspace/miniconda3/envs/wallx/bin/torchrun \
  --nnodes=$NNODES --nproc_per_node=8 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config workspace/zbl_arm/zbl_arm_ee.yml \
  --log_to_file
```

方式 B——平台只给裸节点（自己用主节点 IP 组网；所有节点同一份代码和配置，仅 `--node_rank` 不同）：

```bash
# 节点 0（主节点，<MASTER_IP> 换成该机在集群内网的 IP）
cd /root/workspace/wall-x && \
export https_proxy="http://<USER>:<TOKEN>@proxy.pjlab.org.cn:23128/" \
       http_proxy="$https_proxy" \
       no_proxy="10.0.0.0/8,100.0.0.0/8,.pjlab.org.cn,127.0.0.1,localhost,::1" && \
WANDB_API_KEY=<你的key> \
/root/workspace/miniconda3/envs/wallx/bin/torchrun \
  --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --rdzv_backend=c10d --rdzv_endpoint=<MASTER_IP>:29500 \
  --rdzv_id=zbl-arm-ee \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config workspace/zbl_arm/zbl_arm_ee.yml \
  --log_to_file

# 节点 1：其余完全相同，仅 --node_rank=1
```

多机建议加 `--fsdp_sharding_strategy hybrid_shard`（节点内切分、节点间复制，跨机通信量最小）；首启数据集构建慢可加 `export WALLX_DIST_TIMEOUT_MINUTES=60`。

**Slurm 集群提交示例**（按实际调度器/分区改；本质就是把上面的 torchrun 包进任务脚本）：

```bash
#!/bin/bash
#SBATCH --job-name=zbl-arm-ee
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x-%j.log

export https_proxy="http://<USER>:<TOKEN>@proxy.pjlab.org.cn:23128/"
export http_proxy="$https_proxy"
export no_proxy="10.0.0.0/8,100.0.0.0/8,.pjlab.org.cn,127.0.0.1,localhost,::1"
export WANDB_API_KEY=xxx
MASTER_IP=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NODE_RANK=$((SLURM_NODEID))

torchrun \
  --nnodes="$SLURM_NNODES" --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_IP":29500 \
  --rdzv_id="zbl-arm-$SLURM_JOB_ID" \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config workspace/zbl_arm/zbl_arm_ee.yml \
  --log_to_file
```

### 2.3 训练中

- **实时看曲线**：wandb 项目 `zbl_arm_finetune`，关注 loss / grad_norm / lr；
- **断点续训**：把 yml 的 `checkpoint.resume_from` 从 `.safetensors` 文件改为
  `/path/to/zbl_arm_train/output/<checkpoint目录>`（恢复优化器/步数；支持换卡数 reshard）；
- **调 batch**：A100 80G 显存富余时 `batch_size_per_gpu: 4 → 8`、`gradient_accumulation_steps: 2 → 1`（有效 batch 保持 64）。

### 2.4 冒烟（集群环境就绪后跑一次，1 epoch 快速跑通）

集群开发机（login 节点）一般没有 GPU：在网页平台**起一个 1 卡任务**跑冒烟，"启动命令"填：

```bash
cd /root/workspace/wall-x && \
/root/workspace/miniconda3/envs/wallx/bin/torchrun --nproc_per_node=1 \
  wall_x/trainer/fsdp_trainer/train_fsdp.py \
  --config workspace/zbl_arm/zbl_arm_ee_smoke.yml --log_to_file
```

（冒烟配置 `zbl_arm_ee_smoke.yml` 同样按 §1.4 改路径；单条数据需 `train_test_split: 1.0`。）

---

## 3. 部署完整流程（微调 checkpoint → 真机闭环）

按顺序执行，全部命令已在本机验证通过：

**第 1 步 起推理服务（宿主机，GPU0）**

```bash
cd /home/tangyang/workspace/wall-x-dev-ty_arm
PATH=/home/tangyang/miniconda3/envs/wallx/bin:$PATH \
bash scripts/run_serving.sh \
  --checkpoint-path logs/32 \
  --train-config-path workspace/zbl_arm/zbl_arm_ee.yml \
  --cuda-id 0 --port 32195
```

注意 `run_serving.sh` 必须用 wallx 环境的 PATH 启动（系统 python 缺 yaml）。
换新 checkpoint 只需改 `--checkpoint-path` 重启。

**第 2 步 开环验证（可选但建议，不开真机）**

```bash
cd /home/tangyang/workspace/wall-x-dev-ty_arm
/home/tangyang/miniconda3/envs/wallx/bin/python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32195 \
  --dataset-root /home/tangyang/workspace/zbl_arm_6a_datacol/data/zbl_data/pick_up_test_tube_No_1_from_the_left_rack_and_place_it_in_the_right_rack \
  --train-config workspace/zbl_arm/zbl_arm_ee.yml \
  --episode-indices 0
```

逐维误差对照"零基线"看（详见对话记录的方法论），平移维从基线分离才算学会搬运。

**第 3 步 真机闭环（容器内，机械臂/相机按遥操同样流程先启动）**

```bash
docker exec -it zbl_arm_6a_sdk bash
source /opt/ros/jazzy/setup.bash && source /opt/xr/core/install/setup.bash
python3 /home/xr/workspace/wall-x/workspace/zbl_arm/arm_client.py \
  --task "pick up test tube No. 1 from the left rack and place it in the right rack" \
  --steps 0 --execute --chunk-steps 32
```

执行生命周期：启动打印版本指纹（对版本用）→ 立即匀速归零 → 人工确认门(Enter)→
切笛卡尔控制器 → 策略执行（每块 32 步, 单步限幅 0.8cm/0.08rad, 目标跳变拒绝,
工作空间盒含桌面下限）→ 结束/Ctrl+C/突变终止 → 关节伺服锁定 → 结束后回零。

客户端源码在本仓库 `workspace/zbl_arm/arm_client.py`（容器内副本经 docker cp 同步,
以日志版本指纹核对）；先 `--dry-run` 只打印不动臂。协议与安全机制详见
`CLIENT_DESIGN.md`。

### 开环评测与原图（历史参考）



```bash
bash scripts/run_serving.sh \
  --checkpoint-path /path/to/zbl_arm_train/output/<ckpt> \
  --train-config-path workspace/zbl_arm/zbl_arm_ee.yml \
  --port 32195

# 模型预测 vs 数据集真值
python scripts/draw_openloop_plot.py \
  --uri ws://127.0.0.1:32195 \
  --dataset-root /path/to/zbl_data/<task> \
  --train-config workspace/zbl_arm/zbl_arm_ee.yml \
  --episode-indices 0
```

单臂部署用 `--no-serialize-actions`（原始 32×26 chunk，前 10 维为
[pos3, rot6d, gripper]，后 16 维 padding 忽略），由单臂客户端切片下发。

### 真机闭环客户端（容器内运行）

```bash
docker exec -it zbl_arm_6a_sdk bash
source /opt/ros/jazzy/setup.bash && source /opt/xr/core/install/setup.bash
python3 /home/xr/workspace/wall-x/workspace/zbl_arm/arm_client.py \
  --task "pick up test tube No. 1 from the left rack and place it in the right rack" \
  --steps 0 --execute --chunk-steps 32
```

源码在仓库 `workspace/zbl_arm/arm_client.py`；`--dry-run` 只打印不动臂。详见
`CLIENT_DESIGN.md`（协议约定、安全机制、参数表）。

---

## 4. 注意事项

- 夹爪行程参数必须全数据集一致（当前数据 gripper 上限 ≈1.0 rad），混用不同
  `--gripper-max` 的 episode 会引入分布裂缝。
- 数据按任务一目录一数据集组织；`repo_id` 指到**任务子目录**。多任务训练时逐任务
  配置或先合并数据集，并重算 norm_stats。
- 建议每个任务 50+ episode 后再启动正式训练。
- 多机节点必须同构（每节点卡数相同），异构节点会因 barrier 次数不匹配死锁。
