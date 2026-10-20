#!/usr/bin/env python3
"""zbl 单臂闭环客户端: wall-x WebSocket 推理服务 ↔ 6轴单臂(眼在手单相机).

数据流:
  /arm/end_pose (PoseStamped) ──┐
  /eye_in_hand/**/image_compressed ─┤→ msgpack ws → serve → (32,26) chunk
  /joint_states (退出保持用) ────┘       ↓ 取前10维
  Δpos/Δrot6d 与最新实际位姿复合 → /cart_pose_controller/pose_cmd
  夹爪(绝对rad) → /gripper_controller/commands

状态发送格式(实测约定, 见 wall-x workspace/zbl_arm/CLIENT_DESIGN.md):
  follow2_pos = [x, y, z, roll, pitch, yaw, grip]  (RPY, 与服务器
  euler_to_matrix_zyx_6d 同约定, 由四元数→R→RPY 提取)

用法(容器内):
  python3 arm_client.py --task "pick up test tube No. 1 from the left rack and \
place it in the right rack" --steps 400 --dry-run     # 默认dry-run只打印
  python3 arm_client.py --task "..." --steps 400 --execute   # 真机执行
"""

import argparse
import asyncio
import base64
import math
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import SwitchController

import msgpack
import msgpack_numpy as mn
import websockets


# ---------- 旋转工具(与 wall-x 约定对齐) ----------

def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


def R_to_rpy(R):
    """R = Rz(yaw)Ry(pitch)Rx(roll), 与服务器 euler_to_matrix_zyx_6d 同约定."""
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def rot6d_to_R(v):
    r1 = np.asarray(v[:3], dtype=np.float64)
    r2 = np.asarray(v[3:6], dtype=np.float64)
    # 预测值可能轻微非正交, 重正交化: r1归一, r2减去r1分量后归一
    r1 = r1 / max(np.linalg.norm(r1), 1e-8)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / max(np.linalg.norm(r2), 1e-8)
    r3 = np.cross(r1, r2)
    return np.vstack([r1, r2, r3])


def R_to_rot6d(R):
    return np.concatenate([R[0], R[1]])


def angle_between(R_a, R_b):
    c = np.clip((np.trace(R_a.T @ R_b) - 1.0) / 2.0, -1.0, 1.0)
    return math.acos(c)


# ---------- ROS 节点 ----------

class ArmBridge(Node):

    def __init__(self, args):
        super().__init__('arm_client')
        self.args = args
        self.pose_lock = threading.Lock()
        self._pose = None          # (pos3, R)
        self._img_b64 = None
        self._img_t = 0.0
        self._joints = None
        self._home_ok = False
        self.endpose_sub = self.create_subscription(
            PoseStamped, '/arm/end_pose', self.on_end_pose, 10)
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/eye_in_hand/eye_in_hand/image_raw/image_compressed',
            self.on_image, 10)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.on_joints, 10)
        self.pose_cmd_pub = self.create_publisher(
            PoseStamped, '/cart_pose_controller/pose_cmd', 10)
        self.gripper_cmd_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)
        self.joint_cmd_pub = self.create_publisher(
            Float64MultiArray, '/joint_position_controller/commands', 10)
        self.switch_cli = self.create_client(
            SwitchController, '/controller_manager/switch_controller')

    def on_end_pose(self, msg):
        p = msg.pose.position
        q = msg.pose.orientation
        R = quat_to_R(q.x, q.y, q.z, q.w)
        with self.pose_lock:
            self._pose = (np.array([p.x, p.y, p.z]), R)

    def on_image(self, msg):
        self._img_b64 = base64.b64encode(bytes(msg.data)).decode()
        self._img_t = time.time()

    def on_joints(self, msg):
        pos = np.array(msg.position, dtype=np.float64)
        if msg.name:
            try:
                idx = [list(msg.name).index(f'arm_joint{i}') for i in range(1, 7)]
                pos6 = pos[idx]
            except ValueError:
                pos6 = pos[:6]
        else:
            pos6 = pos[:6]
        self._joints = np.concatenate([pos6, pos[-1:]])  # 6关节+夹爪

    def get_pose(self):
        with self.pose_lock:
            return None if self._pose is None else (self._pose[0].copy(),
                                                    self._pose[1].copy())

    def switch_controllers(self, activate, deactivate=(), timeout=5.0):
        ok = False
        if self.switch_cli.wait_for_service(timeout_sec=timeout):
            req = SwitchController.Request()
            req.activate_controllers = list(activate)
            req.deactivate_controllers = list(deactivate)
            req.strictness = SwitchController.Request.BEST_EFFORT
            fut = self.switch_cli.call_async(req)
            # 节点已由后台线程spin, 这里只轮询future(不能再spin, 会与后台执行器冲突)
            t0 = time.time()
            while not fut.done() and time.time() - t0 < timeout:
                time.sleep(0.05)
            ok = bool(fut.done() and fut.result().ok)
        if ok:
            return True
        # 服务切换失败(常见: 接口被占用) -> CLI 强制切换兜底
        self.get_logger().warn(
            f'服务切换失败, 降级 CLI 强制切换: activate={list(activate)} '
            f'deactivate={list(deactivate)}')
        cmd = ['ros2', 'control', 'switch_controllers']
        if deactivate:
            cmd += ['--deactivate', *deactivate]
        if activate:
            cmd += ['--activate', *activate]
        try:
            import subprocess
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=15)
            if r.returncode == 0:
                return True
            self.get_logger().error(
                f'CLI 切换也失败(rc={r.returncode}): {r.stderr.strip()[:200]}')
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'CLI 切换异常: {e}')
        return False

    def publish_pose(self, pos, R):
        msg = PoseStamped()
        msg.header.frame_id = 'world'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(
            float, pos)
        x, y, z, w = R_to_quat(R)
        msg.pose.orientation.x = x
        msg.pose.orientation.y = y
        msg.pose.orientation.z = z
        msg.pose.orientation.w = w
        self.pose_cmd_pub.publish(msg)

    def publish_gripper(self, value):
        msg = Float64MultiArray()
        msg.data = [float(np.clip(value, 0.0, self.args.gripper_max))]
        self.gripper_cmd_pub.publish(msg)

    def home_slowly(self, speed=0.3, tol=0.02, timeout=30.0):
        """从臂限速匀速回零(master_bridge同款算法).

        预定时长 dur=最大关节角/vmax, 50Hz 发布线性斜坡 q0*(1-t/dur);
        斜坡走完后持续发布0(完整误差破腕部静摩擦), 直到收敛或超时."""
        import rclpy as _rclpy
        if self._joints is None:
            self.get_logger().error('归零: 无关节反馈, 跳过')
            return False
        q0 = np.array(self._joints[:6], dtype=np.float64)
        vmax = max(speed, 0.05)
        dur = float(np.abs(q0).max()) / vmax
        self.get_logger().info(
            f'★ 机械臂匀速回零: 限速{vmax:.2f}rad/s, 预计{dur:.1f}s, '
            f'最大关节{np.degrees(np.abs(q0)).max():.0f}°, 请确保臂周围无人无物!')
        t0 = time.time()
        t_log = t0
        msg = Float64MultiArray()
        while time.time() - t0 < dur + timeout and _rclpy.ok():
            el = time.time() - t0
            s = max(0.0, 1.0 - el / dur) if dur > 0 else 0.0
            msg.data = [float(v * s) for v in q0]
            self.joint_cmd_pub.publish(msg)
            q = np.array(self._joints[:6], dtype=np.float64)
            if s <= 0.0 and np.abs(q).max() < tol:
                self.get_logger().info('归零完成')
                self._home_ok = True
                return True
            if time.time() - t_log > 5.0:
                self.get_logger().info(
                    f'归零中... 最大关节偏差 {np.abs(q).max():.3f}rad '
                    f'(各关节: {np.round(q, 3).tolist()})')
                t_log = time.time()
            time.sleep(0.02)
        q = np.array(self._joints[:6], dtype=np.float64)
        self.get_logger().warn(
            f'归零超时(剩余角度rad: {np.round(q, 3).tolist()}), '
            '可能有关节被阻挡, 请检查. 继续后续流程')
        return False

    def hold_and_release(self):
        """退出: 切回关节控制器并保持当前关节角(位置伺服锁定)."""
        try:
            if self._joints is not None:
                self.switch_controllers(
                    ['joint_position_controller'],
                    ['cart_pose_controller', 'zero_force_dragging_controller'])
                msg = Float64MultiArray()
                msg.data = [float(v) for v in self._joints[:6]]
                # joint_position_controller/commands 为6关节
                pub = self.create_publisher(
                    Float64MultiArray, '/joint_position_controller/commands', 10)
                for _ in range(10):
                    pub.publish(msg)
                    time.sleep(0.02)
                self.get_logger().info('已切换关节伺服并锁定当前位姿')
            else:
                self.switch_controllers([], ['cart_pose_controller'])
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'退出保持失败: {e}')


def fmt(v):
    return np.array2string(np.asarray(v), precision=4, suppress_small=True)


# ---------- 主循环 ----------

async def run(args):
    rclpy.init()
    node = ArmBridge(args)
    import hashlib
    _ver = hashlib.md5(open(__file__, 'rb').read()).hexdigest()[:8]
    node.get_logger().info(f'arm_client 版本指纹: v{_ver} (日志对版本用)')
    def _spin(n):
        try:
            rclpy.spin(n)
        except (ExternalShutdownException, RuntimeError):
            pass

    spin_t = threading.Thread(target=_spin, args=(node,), daemon=True)
    spin_t.start()

    node.get_logger().info(
        f'等待位姿与图像... uri={args.uri} task="{args.task[:40]}..."')
    t0 = time.time()
    while (node.get_pose() is None or node._img_b64 is None
           or node._joints is None):
        if time.time() - t0 > 15:
            node.get_logger().error('15s 未收到 /arm/end_pose 或相机图像, '
                                    '请确认机械臂与相机节点已启动')
            return
        await asyncio.sleep(0.2)

    mode = 'EXECUTE(真机)' if args.execute else 'DRY-RUN(只打印不发布)'
    node.get_logger().info(f'观测就绪, 模式: {mode}, chunk-steps={args.chunk_steps}')

    if args.execute:
        ok = node.switch_controllers(
            ['joint_position_controller'],
            ['cart_pose_controller', 'zero_force_dragging_controller'])
        if not ok:
            node.get_logger().error(
                '关节伺服激活失败(接口被占用), 已中止. 可用 '
                '`ros2 control list_controllers` 检查后手动释放')
            return
        if not args.skip_home:
            node.home_slowly(speed=args.home_speed)
            state = '回零完成' if node._home_ok else '⚠ 归零超时(臂未到零位)'
            input(f'\n=== {state}. 确认臂位姿与桌面安全后按 Enter 开始策略执行, '
                  'Ctrl+C 取消 ===')
        ok = node.switch_controllers(
            ['cart_pose_controller'], ['joint_position_controller'])
        node.get_logger().info(
            '笛卡尔控制器已激活' if ok else
            '⚠ 笛卡尔控制器切换失败(继续以当前控制器发布)')

    stop = {'flag': False}

    def _sig(*_):
        stop['flag'] = True

    signal.signal(signal.SIGINT, _sig)

    step_i = 0
    ws_open = None
    try:
        ws_open = await websockets.connect(args.uri, max_size=64 * 1024 * 1024)
        await ws_open.recv()  # metadata
        while not stop['flag'] and (args.steps <= 0 or step_i < args.steps):
            pose = node.get_pose()
            if pose is None:
                node.get_logger().warn('位姿丢失, 暂停')
                await asyncio.sleep(0.2)
                continue
            if time.time() - node._img_t > 1.0:
                node.get_logger().warn('图像超时(>1s), 暂停')
                await asyncio.sleep(0.2)
                continue
            pos_now, R_now = pose

            r, p, y = R_to_rpy(R_now)
            state7 = [pos_now[0], pos_now[1], pos_now[2], r, p, y,
                      float(node._joints[6]) if len(node._joints) > 6 else 0.0]
            payload = {
                'state': {'follow2_pos': [float(v) for v in state7]},
                'views': {'right_wrist_view': node._img_b64},
                'instruction': args.task,
            }
            t_send = time.time()
            await ws_open.send(msgpack.packb(payload, use_bin_type=True,
                                             default=mn.encode))
            res = msgpack.unpackb(await ws_open.recv(), raw=False,
                                  object_hook=mn.decode)
            chunk = np.asarray(res['predict_action'],
                               dtype=np.float32).reshape(-1, 26)[:, :10]
            rtt = (time.time() - t_send) * 1000
            node.get_logger().info(
                f'chunk#{step_i // args.chunk_steps} rtt={rtt:.0f}ms '
                f'dpos[0]={fmt(chunk[0, :3])}')

            # chunk[k] 是相对观测位姿的累计位移: 必须以 pos_obs/R_obs 为基准,
            # 每步的绝对目标 = 观测位姿 + chunk[k] (闭环纠偏靠每个新chunk重新观测)
            pos_obs, R_obs = pos_now.copy(), R_now.copy()
            prev_pos_cmd = None
            prev_R_cmd = None
            skip_run = 0
            n_exec = min(args.chunk_steps, chunk.shape[0])
            if args.steps > 0:
                n_exec = min(n_exec, args.steps - step_i)
            for k in range(n_exec):
                if stop['flag']:
                    break
                pose = node.get_pose()
                if pose is None:
                    break
                pos_now, R_now = pose
                if not np.isfinite(chunk[k]).all():
                    node.get_logger().error('chunk 含 NaN, 终止并锁定')
                    stop['flag'] = True
                    break
                pos_cmd = pos_obs + chunk[k, :3].astype(np.float64)
                R_cmd = rot6d_to_R(chunk[k, 3:9]) @ R_obs
                # 工作空间盒(含桌面下限): 超界夹回
                b = args.bbox
                pos_cmd[0] = min(max(pos_cmd[0], b[0]), b[1])
                pos_cmd[1] = min(max(pos_cmd[1], b[2]), b[3])
                pos_cmd[2] = min(max(pos_cmd[2], b[4]), b[5])
                # 突变拒绝: 判"相邻目标点的跳变"(整块执行时前瞻距离大是正常的,
                # 目标点瞬移才是突变); 另设宽松绝对距离兜底(臂严重跟不上)
                move = pos_cmd - pos_now
                dist = float(np.linalg.norm(move))
                ang = angle_between(R_now, R_cmd)
                if prev_pos_cmd is None:
                    inc_pos = dist
                    inc_ang = ang
                else:
                    inc_pos = float(np.linalg.norm(pos_cmd - prev_pos_cmd))
                    inc_ang = angle_between(prev_R_cmd, R_cmd)
                if inc_pos > args.max_linear * args.jump_factor or \
                        inc_ang > args.max_angular * args.jump_factor or \
                        dist > 0.15 or ang > 1.0:
                    skip_run += 1
                    node.get_logger().warn(
                        f'  step {step_i}: 突变拒执行 inc={inc_pos:.3f}m/'
                        f'{inc_ang:.3f}rad dist={dist:.3f}m '
                        f'(连续{skip_run}/{args.skip_abort})')
                    if skip_run >= args.skip_abort:
                        node.get_logger().error('连续突变过多, 终止并锁定')
                        stop['flag'] = True
                    step_i += 1
                    await asyncio.sleep(args.period)
                    continue
                skip_run = 0
                prev_pos_cmd, prev_R_cmd = pos_cmd.copy(), R_cmd.copy()
                # 单步限幅(对实际移动量)
                if dist > args.max_linear:
                    pos_cmd = pos_now + move / dist * args.max_linear
                if ang > args.max_angular:
                    f = args.max_angular / ang
                    Rc = R_cmd @ R_now.T
                    theta = f * ang
                    Ac = (Rc - Rc.T) / (2 * math.sin(ang))
                    axis = np.array([Ac[2, 1], Ac[0, 2], Ac[1, 0]])
                    axis = axis / max(np.linalg.norm(axis), 1e-8)
                    K = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
                    Rc = np.eye(3) + math.sin(theta) * K + \
                        (1 - math.cos(theta)) * (K @ K)
                    R_cmd = Rc @ R_now
                grip = float(np.clip(chunk[k, 9], 0.0, args.gripper_max))
                if args.execute:
                    node.publish_pose(pos_cmd, R_cmd)
                    node.publish_gripper(grip)
                else:
                    node.get_logger().info(
                        f'  step {step_i}: dist={dist:.4f} ang={ang:.3f} '
                        f'grip={grip:.3f}')
                step_i += 1
                await asyncio.sleep(args.period)
    except Exception as e:  # noqa: BLE001
        node.get_logger().error(f'主循环异常: {e}')
    finally:
        if ws_open is not None:
            await ws_open.close()
        if args.execute:
            node.hold_and_release()
            if not args.skip_home:
                node.get_logger().info('执行结束, 开始结束后回零')
                node.home_slowly(speed=args.home_speed)
        node.get_logger().info(f'结束: 共执行 {step_i} 步')
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--uri', default='ws://127.0.0.1:32195')
    ap.add_argument('--task', required=True)
    ap.add_argument('--chunk-steps', type=int, default=12,
                    help='每个chunk执行前N步后重新观测(32步=1.6s)')
    ap.add_argument('--steps', type=int, default=400,
                    help='总执行步数(20fps, 400步=20s); 0=不限, 直到Ctrl+C')
    ap.add_argument('--period', type=float, default=0.05)
    ap.add_argument('--max-linear', type=float, default=0.008,
                    help='单步最大平移(m)')
    ap.add_argument('--max-angular', type=float, default=0.08,
                    help='单步最大旋转(rad)')
    ap.add_argument('--jump-factor', type=float, default=8.0,
                    help='目标位移超过 max-linear*jump_factor 判定为突变, 拒绝执行')
    ap.add_argument('--skip-abort', type=int, default=6,
                    help='连续拒绝步数达到该值则终止并锁定')
    ap.add_argument('--bbox', type=float, nargs=6,
                    default=[-0.06, 0.19, -0.03, 0.32, -0.045, 0.115],
                    metavar=('X_MIN', 'X_MAX', 'Y_MIN', 'Y_MAX', 'Z_MIN', 'Z_MAX'),
                    help='工作空间盒(含桌面下限), 目标超出则夹到边界')
    ap.add_argument('--gripper-max', type=float, default=1.0)
    ap.add_argument('--home-speed', type=float, default=0.3,
                    help='归零关节速度(rad/s)')
    ap.add_argument('--skip-home', action='store_true',
                    help='跳过执行前的缓慢归零')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--dry-run', action='store_true', default=True)
    g.add_argument('--execute', action='store_true')
    asyncio.run(run(ap.parse_args()))


if __name__ == '__main__':
    main()
