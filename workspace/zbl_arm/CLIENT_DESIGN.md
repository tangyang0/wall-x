# zbl 单臂闭环客户端设计（Wall-X 部署第 3 步）

目标：在 zbl 6 轴单臂上闭环运行微调后的 wall-oss 模型（logs/32）。
骨架复用 `zbl_arm_6a_datacol/master_bridge.py`（ROS2 订阅/发布、限幅、参数体系）。

## 已验证的协议（来自 draw_openloop_plot.py 实测）

- 传输：WebSocket + msgpack（`msgpack.packb(payload, use_bin_type=True)`）；
- 连接后先收一次 metadata（msgpack dict）；
- 每步发送：`{"state": {...}, "views": {相机名: b64 JPEG}, "instruction": 任务文本}`；
- 相机名 = train config `key_mappings.camera` 的值侧（我们：`right_wrist_view`）；
- 收到：32×10 动作 chunk（padding 16 维已被服务端剥掉），布局
  `[Δpos3, Δrot6d6, gripper1]`（relative 键 + 绝对夹爪），20fps 语义。

## 状态输入约定（已实测敲定，无需改服务端代码）

`robot.py` 的 `robot_action_key_mapping` 按 `[pos3, RPY3, grip1]` 切 `follow2_pos`，
旋转经 `euler_to_matrix_zyx_6d` 自动重建 6D。**客户端必须发 7 维欧拉布局**：

```python
rpy = Rotation.from_matrix(R).as_euler("XYZ")   # R 由 6D 前两行 + 叉积重建
follow2_pos = [x, y, z, roll, pitch, yaw, grip]
```

实测证据（2026-10-18）：状态位置 +0.5m 扰动 → 预测变化（max diff 0.0195），
状态通路确认畅通；模型对本体感受权重低（视觉主导），属模型侧特性而非管线问题。
原始返回 chunk 为 **32×26**（含 padding），客户端取 `[:, :10]`。

## 客户端数据流

```
/joint_states(7)  /endpose(topic, 位姿+四元数)   /eye_in_hand/image_raw
        │                │                            │
        │          quat→6D/→RPY(状态用)                │ b64 JPEG
        └────────────────┴────────────┬───────────────┘
                                    msgpack ws 发送
                                        │
                                  serve:32195 (32,10)
                                        │
                 Δpos 与当前 pos 相加；Δrot6d 与当前姿态复合
                 (R_abs = R_delta @ R_state，参照 loader 的
                  compose_state_and_delta_to_abs_6d，numpy 重写)
                 夹爪值直接下发（绝对，0~gripper-max rad）
                                        │
              限幅(工作空间/单步最大位移/最大角速度) → /cart_pose_controller + /gripper_controller/commands
```

- 位姿来源：直接订阅端位姿话题（录制时 on_end_pose 同源），无需 FK；
- 6D↔四元数互转：lerobot_writer.quat_to_6d 的逆（前两行→R→第三行叉积）；
- chunk 消费策略：每 32 步 = 1.6s；建议每 chunk 只执行前 N 步（N≈8~16）就重新
  观测请求（动作更贴合实际状态），N 做成参数；
- 控制频率：20fps 与训练对齐（插值可用 vla-pipeline 的 multiplier 思路）。

## 安全（参考 vla-pipeline 的 clamp 体系）

- 工作空间盒限幅（从 50 条数据的 q01/q99 外扩 10% 初始化）；
- 单步最大平移 0.02m / 最大旋转 0.15rad（数据分布 p99 校准）；
- 夹爪范围 [0, gripper_max]；
- 生成式急停：收不到新 chunk 超时 >2s → 保持当前位姿（位置伺服锁定）；
- 首跑必须 `--dry-run`：只打印动作不发布；再 `--steps N` 限量执行。

## 参数与代码位置

- 代码：`workspace/zbl_arm/arm_client.py`（本仓库 dev/ty_arm；容器内运行副本在
  `/home/xr/workspace/wall-x/workspace/zbl_arm/arm_client.py`，宿主机改后 docker cp 同步）；
- 参数：`--uri ws://127.0.0.1:32195 --task "pick up test tube No. 1 ..." --side right
  --chunk-steps 12 --dry-run --steps N --max-linear 0.02 --max-angular 0.15`；
- 训练任务文本必须与录制时逐字一致（从 tasks.parquet 复制）。

## 上真机前的验收清单

1. [ ] 服务端 6D 映射补丁生效（draw_openloop L1 随状态变化）
2. [ ] dry-run：chunk 形状/数值范围/任务文本正确，动作连续无跳变
3. [ ] --steps 32 真机小步执行，人在急停旁
4. [ ] 全程录像 + episode 数据记录（复用数采管线可回溯）
