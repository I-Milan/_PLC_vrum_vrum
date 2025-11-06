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

        # --- Alap paraméterek (a te params.yaml-od alapján) ---
        self.side = self.declare_parameter('side', 'left').get_parameter_value().string_value
        assert self.side in ('left', 'right'), 'side param must be "left" or "right"'
        self.sign = 1.0 if self.side == 'left' else -1.0

        self.desired = float(self.declare_parameter('desired_distance', 1.5).value)
        self.L = float(self.declare_parameter('lookahead_distance', 1.0).value)

        self.kp = float(self.declare_parameter('kp', 14.0).value)
        self.ki = float(self.declare_parameter('ki', 0.0).value)
        self.kd = float(self.declare_parameter('kd', 0.09).value)

        self.v_lin = float(self.declare_parameter('linear_speed', 0.6).value)
        self.v_lin_min = float(self.declare_parameter('linear_speed_min', 0.15).value)
        self.w_lim = float(self.declare_parameter('angular_speed_limit', 4.8).value)

        self.slow_down_dist = float(self.declare_parameter('slow_down_distance', 0.8).value)
        self.stop_dist = float(self.declare_parameter('stop_distance', 0.35).value)

        self.median_window = int(self.declare_parameter('median_window', 5).value)
        self.range_clip_max = float(self.declare_parameter('range_clip_max', 10.0).value)
        self.valid_min = float(self.declare_parameter('valid_min', 0.03).value)

        self.angle_a_deg = float(self.declare_parameter('angle_a_deg', 90.0).value)
        self.angle_b_deg = float(self.declare_parameter('angle_b_deg', 45.0).value)
        self.forward_half_fov_deg = float(self.declare_parameter('forward_half_fov_deg', 15.0).value)

        # LPF / reversal boost (megtartjuk, de nem piszkáljuk)
        self.steer_lpf_alpha = float(self.declare_parameter('steer_lpf_alpha', 0.5).value)
        self.steer_alpha_min = float(self.declare_parameter('steer_alpha_min', 0.3).value)
        self.steer_alpha_max = float(self.declare_parameter('steer_alpha_max', 0.95).value)
        self.reversal_boost_factor = float(self.declare_parameter('reversal_boost_factor', 1.3).value)
        self.reversal_bypass_time_s = float(self.declare_parameter('reversal_bypass_time_s', 0.25).value)

        # Goalpoint mód
        self.follow_angle_deg = float(self.declare_parameter('follow_angle_deg', 45.0).value)
        self.lookahead_forward = float(self.declare_parameter('lookahead_forward', 1.00).value)
        self.k_goal = float(self.declare_parameter('k_goal', 1.0).value)
        self.use_goalpoint = bool(self.declare_parameter('use_goalpoint', True).value)

        # Oldalváltás
        self.side_switch_enable = bool(self.declare_parameter('side_switch_enable', True).value)
        self.side_presence_sector_deg = float(self.declare_parameter('side_presence_sector_deg', 15.0).value)
        self.side_presence_max = float(self.declare_parameter('side_presence_max', 2.5).value)
        self.side_switch_hold_s = float(self.declare_parameter('side_switch_hold_s', 1.0).value)

        # Származtatott szögek az aktuális oldalhoz
        self.angle_a = math.radians(self.sign * self.angle_a_deg)  # ~90°
        self.angle_b = math.radians(self.sign * self.angle_b_deg)  # ~45°

        # Dinamikus cél (oldalváltáshoz) – induláskor a névleges desired
        self.desired_current = self.desired
        self._desired_min = 0.5   # ésszerű bilincsek (nem paramozzuk túl)
        self._desired_max = 2.0

        # Min-kanyar “mentőöv”
        self._rescue_r45_ratio = 0.9      # 45° nagy, ha > 0.9 * side_presence_max
        self._rescue_angle = 0.10         # rad (~5.7°) alatt kötelező fordulás
        self._rescue_gain = 0.50          # w_lim 50%-a

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

    # --------- Szög illesztése a scan tartományába (KRITIKUS JAVÍTÁS!) ---------
    def _wrap_angle_to_scan(self, angle_rad: float, scan: LaserScan) -> float:
        """Visszaad egy olyan szöget, ami a [angle_min, angle_max] tartományba esik,
        2π-s periodicitással. Így a -90° is helyesen a 270°-nak megfelelő indexre kerül,
        ha a scan 0..2π tartományban van."""
        amin, amax = scan.angle_min, scan.angle_max
        width = amax - amin
        if width == 0.0:
            return amin
        # normalizálás 0..width közé
        a = (angle_rad - amin) % (2.0 * math.pi)
        # ha a szenzor tartománya kisebb/nagyobb, toljuk vissza a saját sávjába
        if a > width:
            a -= 2.0 * math.pi
        return amin + a

    def _angle_to_index(self, scan: LaserScan, angle_rad: float) -> int:
        # <<< EZ AZ ÚJ: előbb wrap, aztán index >>>
        a_wrapped = self._wrap_angle_to_scan(angle_rad, scan)
        inc = scan.angle_increment
        if inc == 0.0:
            return 0
        idx = int(round((a_wrapped - scan.angle_min) / inc))
        return clamp(idx, 0, len(scan.ranges) - 1)

    # --------- LiDAR segéd ---------
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

    def _min_forward(self, scan: LaserScan) -> float:
        half = math.radians(self.forward_half_fov_deg)
        idx_lo = self._angle_to_index(scan, -half)
        idx_hi = self._angle_to_index(scan, +half)
        lo, hi = (min(idx_lo, idx_hi), max(idx_lo, idx_hi))
        mins = self.range_clip_max
        for i in range(lo, hi + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and r >= self.valid_min:
                mins = min(mins, min(r, self.range_clip_max))
        return mins

    # --------- Geometria: két pont módszer (backup) ---------
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

    # --------- Goalpoint 45° alapján ---------
    def _goalpoint_from_45(self, scan: LaserScan):
        a = math.radians(self.sign * self.follow_angle_deg)
        r = self._get_range_at(scan, a, half_window=1)
        if not math.isfinite(r) or r >= self.range_clip_max:
            return None
        px, py = r * math.cos(a), r * math.sin(a)
        norm = math.hypot(px, py)
        if norm < 1e-6:
            return None
        nx, ny = -px / norm, -py / norm
        gx = px + self.desired_current * nx + self.lookahead_forward
        gy = py + self.desired_current * ny
        return gx, gy

    # --------- Oldalváltás + dinamikus cél beállítás ---------
    def _maybe_switch_side(self, scan: LaserScan, now_s: float):
        if not self.side_switch_enable:
            return
        side_center = math.radians(self.sign * 90.0)
        d_avg = self._avg_range_in_sector(scan, side_center, self.side_presence_sector_deg)

        # Kiegészítés: 45° is számítson "fal-eltűnésnek"
        r45 = self._get_range_at(scan, math.radians(self.sign * 45.0), 1)
        if not math.isfinite(r45):
            r45 = self.range_clip_max

        need_switch = (d_avg > self.side_presence_max) or (r45 > 0.9 * self.side_presence_max)

        if need_switch and (now_s - self.last_switch_t) > self.side_switch_hold_s:
            old_side = self.side
            self.sign *= -1.0
            self.side = 'left' if self.sign > 0 else 'right'
            self.last_switch_t = now_s

            # új szögek az új oldalhoz
            self.angle_a = math.radians(self.sign * self.angle_a_deg)
            self.angle_b = math.radians(self.sign * self.angle_b_deg)

            # PID reset
            self.int_err = 0.0
            self.prev_err = 0.0

            # Dinamikus cél: új oldal 90°-ából (ésszerű bilincs)
            d_new = self._avg_range_in_sector(scan, math.radians(self.sign * 90.0), self.side_presence_sector_deg)
            if not math.isfinite(d_new):
                d_new = self.desired
            self.desired_current = clamp(d_new, self._desired_min, self._desired_max)

            self.get_logger().info(
                f'[SWITCH] {old_side} -> {self.side}; d90={d_avg:.2f} r45={r45:.2f} desired_cur={self.desired_current:.2f}'
            )

    # --------- Dinamikus LPF + reversal boost ---------
    def _filter_and_clamp_w(self, w, now):
        if (w * self._last_w) < 0.0 and abs(w) > 0.5 * self.w_lim:
            self._reversal_until = now + self.reversal_bypass_time_s
            w *= self.reversal_boost_factor

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

        # debug változók
        theta = 0.0
        a = float('nan')
        b = float('nan')
        d_t = float('nan')
        err = 0.0

        # oldalváltás (ha kell)
        self._maybe_switch_side(scan, now)

        # előre biztonság
        min_fwd = self._min_forward(scan)

        # goalpoint
        w_from_goal = None
        if self.use_goalpoint:
            gp = self._goalpoint_from_45(scan)
            if gp is not None:
                gx, gy = gp
                ang_goal = math.atan2(gy, gx)
                w_from_goal = self.k_goal * ang_goal

                # Min-kanyar mentőöv
                r45_now = self._get_range_at(scan, math.radians(self.sign * 45.0), 1)
                if not math.isfinite(r45_now):
                    r45_now = self.range_clip_max
                if (r45_now > self._rescue_r45_ratio * self.side_presence_max) and (abs(ang_goal) < self._rescue_angle):
                    w_from_goal = self.sign * self._rescue_gain * self.w_lim

        if w_from_goal is not None:
            # célpontos ág
            w_raw = w_from_goal

            v = self.v_lin
            gamma = 2.0
            turn_ratio = clamp(abs(w_raw) / max(self.w_lim, 1e-6), 0.0, 1.0)
            turn_slowdown = max(0.1, 1.0 - (turn_ratio ** gamma))
            v = max(self.v_lin_min, self.v_lin * turn_slowdown)
            if min_fwd < self.slow_down_dist:
                v = max(self.v_lin_min, v * 0.35)
            if min_fwd < self.stop_dist:
                v = 0.0
                w = (-self.sign) * 0.8 * self.w_lim
                self._w_filt = w
            else:
                w = self._filter_and_clamp_w(w_raw, now)

        else:
            # backup: két-sugaras PID
            err_raw, theta, a, b, d_t = self._compute_error(scan)
            err = self.med_err.push(err_raw)

            prev_err_prior = self.prev_err
            self.int_err += err * dt
            self.int_err = clamp(self.int_err, -1.0, 1.0)
            if err * prev_err_prior < 0.0 or min_fwd < self.slow_down_dist:
                self.int_err = 0.0
            der = (err - prev_err_prior) / dt if dt > 0.0 else 0.0
            self.prev_err = err

            w_raw = self.kp * err + self.ki * self.int_err + self.kd * der
            w_raw = -self.sign * w_raw

            v = self.v_lin
            gamma = 2.0
            turn_ratio = clamp(abs(w_raw) / max(self.w_lim, 1e-6), 0.0, 1.0)
            turn_slowdown = max(0.1, 1.0 - (turn_ratio ** gamma))
            v = max(self.v_lin_min, self.v_lin * turn_slowdown)
            if min_fwd < self.slow_down_dist:
                v = max(self.v_lin_min, v * 0.35)
            if min_fwd < self.stop_dist:
                v = 0.0
                w = (-self.sign) * 0.8 * self.w_lim
                self._w_filt = w
            else:
                w = self._filter_and_clamp_w(w_raw, now)

        # publikálás
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        # debug (opcionális)
        try:
            self.get_logger().debug(
                f"side={self.side} err={err:.3f} theta={math.degrees(theta):.1f} "
                f"a={a:.2f} b={b:.2f} d_t={d_t:.2f} v={v:.2f} w={w:.2f} desired_cur={self.desired_current:.2f}"
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
