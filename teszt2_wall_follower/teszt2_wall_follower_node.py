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

        # --- Alap paraméterek ---
        self.side = self.declare_parameter('side', 'left').get_parameter_value().string_value
        assert self.side in ('left', 'right'), 'side param must be "left" or "right"'
        self.sign = 1.0 if self.side == 'left' else -1.0

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

        # LPF (fix) + Dinamikus LPF / reversal boost paramok
        self.steer_lpf_alpha = float(self.declare_parameter('steer_lpf_alpha', 0.5).value)  # régi, kompatibilitás
        self.steer_alpha_min = float(self.declare_parameter('steer_alpha_min', 0.3).value)
        self.steer_alpha_max = float(self.declare_parameter('steer_alpha_max', 0.95).value)
        self.reversal_boost_factor = float(self.declare_parameter('reversal_boost_factor', 1.3).value)
        self.reversal_bypass_time_s = float(self.declare_parameter('reversal_bypass_time_s', 0.25).value)

        # 45°-os célpont mód (opcionális, a két-sugaras PID mellé/fölé)
        self.follow_angle_deg = float(self.declare_parameter('follow_angle_deg', 45.0).value)
        self.lookahead_forward = float(self.declare_parameter('lookahead_forward', 0.30).value)
        self.k_goal = float(self.declare_parameter('k_goal', 1.2).value)
        self.use_goalpoint = bool(self.declare_parameter('use_goalpoint', True).value)

        # Oldal „eltűnés” detektálás és automatikus váltás
        self.side_switch_enable = bool(self.declare_parameter('side_switch_enable', True).value)
        self.side_presence_sector_deg = float(self.declare_parameter('side_presence_sector_deg', 15.0).value)
        self.side_presence_max = float(self.declare_parameter('side_presence_max', 2.5).value)
        self.side_switch_hold_s = float(self.declare_parameter('side_switch_hold_s', 1.0).value)

        # Származtatott
        self.angle_a = math.radians(self.sign * self.angle_a_deg)
        self.angle_b = math.radians(self.sign * self.angle_b_deg)

        # Állapotok
        self.int_err = 0.0
        self.prev_err = 0.0
        self.prev_t = None
        self._w_filt = 0.0
        self._last_w = 0.0
        self._reversal_until = 0.0
        self.last_switch_t = 0.0

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, qos)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.med_err = MedianFilter(self.median_window)

        self.get_logger().info(
            f'teszt2_wall_follower running. side={self.side}, desired={self.desired} m'
        )

    # --------- LiDAR segéd ---------
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

    # --------- Geometria: két pont módszer ---------
    def _compute_error(self, scan: LaserScan):
        """
        a = 90° oldaltáv, b = 45° előre-oldal, beta = |a-b|
        theta = atan2(a*cos(beta) - b, a*sin(beta))
        d_t   = b*cos(theta) - L*sin(theta)
        err   = desired - d_t
        """
        beta = abs(self.angle_a - self.angle_b)
        a = self._get_range_at(scan, self.angle_a, half_window=1)
        b = self._get_range_at(scan, self.angle_b, half_window=1)

        a = min(a, self.range_clip_max)
        b = min(b, self.range_clip_max)

        theta = math.atan2(a * math.cos(beta) - b, a * math.sin(beta))
        d_t = b * math.cos(theta) - self.L * math.sin(theta)
        err = self.desired - d_t
        return err, theta, a, b, d_t

    # --------- Célpont 45° alapján (opcionális kormány) ---------
    def _avg_range_in_sector(self, scan: LaserScan, center_angle: float, half_width_deg: float) -> float:
        hw = math.radians(half_width_deg)
        i0 = self._angle_to_index(scan, center_angle - hw)
        i1 = self._angle_to_index(scan, center_angle + hw)
        lo, hi = (min(i0, i1), max(i0, i1))
        vals = []
        for i in range(lo, hi + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and r >= self.valid_min:
                vals.append(min(r, self.range_clip_max))
        return sum(vals) / len(vals) if vals else self.range_clip_max

    def _goalpoint_from_45(self, scan: LaserScan):
        """
        45°-on mért pont -> célpont:
          P = r [cos(a), sin(a)]
          n = -P/|P|  (fal normálja a robot felé)
          G = P + desired*n + [lookahead_forward, 0]
        """
        a = math.radians(self.sign * self.follow_angle_deg)
        r = self._get_range_at(scan, a, half_window=1)
        if not math.isfinite(r) or r >= self.range_clip_max:
            return None
        px, py = r * math.cos(a), r * math.sin(a)
        norm = math.hypot(px, py)
        if norm < 1e-6:
            return None
        nx, ny = -px / norm, -py / norm
        gx = px + self.desired * nx + self.lookahead_forward
        gy = py + self.desired * ny
        return gx, gy

    def _maybe_switch_side(self, scan: LaserScan, now_s: float):
        """Ha az aktuális oldalon 'nincs fal' (90° környéke messze), és letelt a hold, válts oldalt."""
        if not self.side_switch_enable:
            return
        side_center = math.radians(self.sign * 90.0)
        d_avg = self._avg_range_in_sector(scan, side_center, self.side_presence_sector_deg)
        if d_avg > self.side_presence_max and (now_s - self.last_switch_t) > self.side_switch_hold_s:
            self.sign *= -1.0
            self.side = 'left' if self.sign > 0 else 'right'
            self.last_switch_t = now_s
            # új szögek a másik oldalhoz
            self.angle_a = math.radians(self.sign * self.angle_a_deg)
            self.angle_b = math.radians(self.sign * self.angle_b_deg)
            # PID reset
            self.int_err = 0.0
            self.prev_err = 0.0
            self.get_logger().info(f'Fal eltűnt – oldalváltás: {self.side}')

    # --------- Dinamikus LPF + reversal boost ---------
    def _filter_and_clamp_w(self, w, now):
        # reversal detektálás: előjelváltás nagy kéréssel
        if (w * self._last_w) < 0.0 and abs(w) > 0.5 * self.w_lim:
            self._reversal_until = now + self.reversal_bypass_time_s
            w *= self.reversal_boost_factor

        # dinamikus alpha: nagy |w| -> nagy alpha (kevesebb simítás)
        if now < self._reversal_until:
            alpha_dyn = self.steer_alpha_max
        else:
            ratio = clamp(abs(w) / max(self.w_lim, 1e-6), 0.0, 1.0)
            alpha_dyn = self.steer_alpha_min + (self.steer_alpha_max - self.steer_alpha_min) * ratio

        self._w_filt = (1.0 - alpha_dyn) * self._w_filt + alpha_dyn * w
        w_out = clamp(self._w_filt, -self.w_lim, self.w_lim)
        self._last_w = w_out
        return w_out

    # --------- Fő callback ---------
    def scan_cb(self, scan: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.prev_t is None else max(1e-3, now - self.prev_t)
        self.prev_t = now

        # debug változók alapérték
        theta = 0.0
        a = float('nan')
        b = float('nan')
        d_t = float('nan')
        err = 0.0

        # esetleges oldalváltás
        self._maybe_switch_side(scan, now)

        # biztonság előre
        min_fwd = self._min_forward(scan)

        # célpont mód (ha engedélyezett és van érvényes célpont)
        w_from_goal = None
        if self.use_goalpoint:
            gp = self._goalpoint_from_45(scan)
            if gp is not None:
                gx, gy = gp
                ang_goal = math.atan2(gy, gx)
                w_from_goal = self.k_goal * ang_goal

        if w_from_goal is not None:
            # --- Célpontos ág ---
            w_raw = w_from_goal

            # sebesség menedzsment
            v = self.v_lin
            gamma = 2.0
            turn_ratio = clamp(abs(w_raw) / max(self.w_lim, 1e-6), 0.0, 1.0)
            turn_slowdown = max(0.1, 1.0 - (turn_ratio ** gamma))
            v = max(self.v_lin_min, self.v_lin * turn_slowdown)
            if turn_ratio > 0.85:
                v = min(v, 0.05)

            if min_fwd < self.slow_down_dist:
                v = max(self.v_lin_min, v * 0.35)

            if min_fwd < self.stop_dist:
                v = 0.0
                w = (-self.sign) * 0.8 * self.w_lim
                self._w_filt = w  # azonnali
            else:
                w = self._filter_and_clamp_w(w_raw, now)

        else:
            # --- Két-sugaras PID ág ---
            err_raw, theta, a, b, d_t = self._compute_error(scan)
            err = self.med_err.push(err_raw)

            prev_err_prior = self.prev_err

            # integrál + bilincs
            self.int_err += err * dt
            self.int_err = clamp(self.int_err, -1.0, 1.0)

            # előjel-váltásnál vagy közel akadálynál engedjük el
            if err * prev_err_prior < 0.0 or min_fwd < self.slow_down_dist:
                self.int_err = 0.0

            der = (err - prev_err_prior) / dt if dt > 0.0 else 0.0
            self.prev_err = err

            w_raw = self.kp * err + self.ki * self.int_err + self.kd * der
            w_raw = -self.sign * w_raw

            # deadband
            err_deadband = 0.02  # ~2 cm
            if abs(err) < err_deadband:
                w_raw = 0.0
                self.int_err *= 0.5

            # sebesség menedzsment
            v = self.v_lin
            gamma = 2.0
            turn_ratio = clamp(abs(w_raw) / max(self.w_lim, 1e-6), 0.0, 1.0)
            turn_slowdown = max(0.1, 1.0 - (turn_ratio ** gamma))
            v = max(self.v_lin_min, self.v_lin * turn_slowdown)
            if turn_ratio > 0.85:
                v = min(v, 0.05)

            if min_fwd < self.slow_down_dist:
                v = max(self.v_lin_min, v * 0.35)

            if min_fwd < self.stop_dist:
                v = 0.0
                w = (-self.sign) * 0.8 * self.w_lim
                self._w_filt = w  # azonnali
            else:
                w = self._filter_and_clamp_w(w_raw, now)

        # publikálás
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        # debug
        try:
            self.get_logger().debug(
                f"side={self.side} err={err:.3f} theta={math.degrees(theta):.1f} "
                f"a={a:.2f} b={b:.2f} d_t={d_t:.2f} min_fwd={min_fwd:.2f} v={v:.2f} w={w:.2f}"
            )
        except Exception:
            pass


def main():
    rclpy.init()
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
