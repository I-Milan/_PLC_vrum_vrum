import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class MedianFilter:
    def __init__(self, window):
        self.window = window
        self.buf = deque(maxlen=window)

    def push(self, x):
        self.buf.append(x)
        if len(self.buf) == 0:
            return x
        return sorted(self.buf)[len(self.buf)//2]


class WallFollower(Node):
    def __init__(self):
        super().__init__('teszt2_wall_follower')

        # --- Params ---
        self.side = self.declare_parameter('side', 'left').get_parameter_value().string_value
        assert self.side in ('left', 'right'), 'side param must be "left" or "right"'

        self.desired = float(self.declare_parameter('desired_distance', 0.6).value)
        self.L = float(self.declare_parameter('lookahead_distance', 0.5).value)

        self.kp = float(self.declare_parameter('kp', 1.8).value)
        self.ki = float(self.declare_parameter('ki', 0.0).value)
        self.kd = float(self.declare_parameter('kd', 0.22).value)

        self.v_lin = float(self.declare_parameter('linear_speed', 0.5).value)
        self.v_lin_min = float(self.declare_parameter('linear_speed_min', 0.15).value)
        self.w_lim = float(self.declare_parameter('angular_speed_limit', 1.8).value)

        self.slow_down_dist = float(self.declare_parameter('slow_down_distance', 0.8).value)
        self.stop_dist = float(self.declare_parameter('stop_distance', 0.35).value)

        self.median_window = int(self.declare_parameter('median_window', 5).value)
        self.range_clip_max = float(self.declare_parameter('range_clip_max', 10.0).value)
        self.valid_min = float(self.declare_parameter('valid_min', 0.03).value)

        self.angle_a_deg = float(self.declare_parameter('angle_a_deg', 90.0).value)
        self.angle_b_deg = float(self.declare_parameter('angle_b_deg', 45.0).value)
        self.forward_half_fov_deg = float(self.declare_parameter('forward_half_fov_deg', 15.0).value)

        self.sign = 1.0 if self.side == 'left' else -1.0
        self.angle_a = math.radians(self.sign * self.angle_a_deg)
        self.angle_b = math.radians(self.sign * self.angle_b_deg)

        self.steer_lpf_alpha = float(self.declare_parameter('steer_lpf_alpha', 0.5).value)

        self.int_err = 0.0
        self.prev_err = 0.0
        self.prev_t = None
        self._w_filt = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, qos)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.med_err = MedianFilter(self.median_window)

        self.get_logger().info(f'teszt2_wall_follower running. side={self.side}, desired={self.desired} m')

    def _angle_to_index(self, scan: LaserScan, angle_rad: float) -> int:
        inc = scan.angle_increment
        if inc == 0.0:
            return 0
        if inc > 0.0:
            idx = int(round((angle_rad - scan.angle_min) / inc))
        else:
            idx = int(round((scan.angle_min - angle_rad) / (-inc)))
        return clamp(idx, 0, len(scan.ranges) - 1)

    def _get_range_at(self, scan: LaserScan, angle_rad: float, half_window: int = 1) -> float:
        i_center = self._angle_to_index(scan, angle_rad)
        lo = clamp(i_center - half_window, 0, len(scan.ranges) - 1)
        hi = clamp(i_center + half_window, 0, len(scan.ranges) - 1)
        vals = []
        for i in range(lo, hi + 1):
            r = scan.ranges[i]
            if math.isfinite(r):
                if r >= self.valid_min:
                    vals.append(min(r, self.range_clip_max))
            else:
                vals.append(self.range_clip_max)
        if not vals:
            return self.range_clip_max
        return sum(vals) / len(vals)

    def _min_forward(self, scan: LaserScan) -> float:
        half = math.radians(self.forward_half_fov_deg)
        idx_lo = self._angle_to_index(scan, -half)
        idx_hi = self._angle_to_index(scan, +half)
        lo, hi = (idx_lo, idx_hi) if idx_lo <= idx_hi else (idx_hi, idx_lo)
        mins = self.range_clip_max
        for i in range(lo, hi + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and r >= self.valid_min:
                mins = min(mins, min(r, self.range_clip_max))
        return mins

    def _compute_error(self, scan: LaserScan):
        beta = abs(self.angle_a - self.angle_b)
        a = self._get_range_at(scan, self.angle_a, half_window=1)
        b = self._get_range_at(scan, self.angle_b, half_window=1)

        a = min(a, self.range_clip_max)
        b = min(b, self.range_clip_max)

        theta = math.atan2(a * math.cos(beta) - b, a * math.sin(beta))
        d_t = b * math.cos(theta) - self.L * math.sin(theta)

        err = self.desired - d_t
        return err, theta, a, b, d_t

    def scan_cb(self, scan: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.prev_t is None else max(1e-3, now - self.prev_t)
        self.prev_t = now

        min_fwd = self._min_forward(scan)

        err_raw, theta, a, b, d_t = self._compute_error(scan)
        err = self.med_err.push(err_raw)

        prev_err_prior = self.prev_err
        self.int_err += err * dt

        i_max = 1.0
        self.int_err = clamp(self.int_err, -i_max, i_max)

        if err * prev_err_prior < 0.0 or min_fwd < self.slow_down_dist:
            self.int_err = 0.0

        der = (err - prev_err_prior) / dt if dt > 0.0 else 0.0
        self.prev_err = err

        w = self.kp * err + self.ki * self.int_err + self.kd * der
        w = -self.sign * w

        err_deadband = 0.02
        if abs(err) < err_deadband:
            w = 0.0
            self.int_err *= 0.5

        v = self.v_lin

        turn_slowdown = max(0.2, 1.0 - (abs(w) / max(self.w_lim, 1e-6)))
        v = max(self.v_lin_min, v * turn_slowdown)

        if min_fwd < self.slow_down_dist:
            v = max(self.v_lin_min, v * 0.35)

        if min_fwd < self.stop_dist:
            v = 0.0
            w = (-self.sign) * 0.8 * self.w_lim
            self._w_filt = w
        else:
            alpha = clamp(self.steer_lpf_alpha, 0.0, 1.0)
            self._w_filt = (1.0 - alpha) * self._w_filt + alpha * w
            w = self._w_filt
            w = clamp(w, -self.w_lim, self.w_lim)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

    def main():
        rclpy.init()
        node = WallFollower()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        node.destroy_node()
        rclpy.shutdown()

