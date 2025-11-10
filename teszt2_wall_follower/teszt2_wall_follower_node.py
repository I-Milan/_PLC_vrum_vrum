import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time


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

        # LPF / boost
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

        # Oldalváltás – RÉGI (90°) logika opcionális
        self.presence_switch_enable = bool(self.declare_parameter('presence_switch_enable', False).value)
        self.side_presence_sector_deg = float(self.declare_parameter('side_presence_sector_deg', 15.0).value)
        self.side_presence_max = float(self.declare_parameter('side_presence_max', 2.5).value)

        # Oldalváltás – ÚJ: goalpoint-ugrás alapján
        self.gp_jump_enable = bool(self.declare_parameter('gp_jump_enable', True).value)
        self.gp_jump_thresh = float(self.declare_parameter('gp_jump_thresh', 1.8).value)   # m
        self.gp_jump_frames = int(self.declare_parameter('gp_jump_frames', 2).value)       # egymás utáni frame-ek

        self.side_switch_hold_s = float(self.declare_parameter('side_switch_hold_s', 1.0).value)

        # Saroktartalék oldalváltás után
        self.post_switch_expand_s = float(self.declare_parameter('post_switch_expand_s', 5.0).value)
        self.post_switch_expand_add = float(self.declare_parameter('post_switch_expand_add', 0.30).value)
        self._expand_until = 0.0

        # Desired visszacsengés (visszahúzás a végső desired felé)
        self.desired_relax_tau_s = float(self.declare_parameter('desired_relax_tau_s', 2.0).value)

        # Szögek az aktuális oldalhoz
        self.angle_a = math.radians(self.sign * self.angle_a_deg)  # 90°
        self.angle_b = math.radians(self.sign * self.angle_b_deg)  # 45°

        # Dinamikus cél (váltáshoz)
        self.desired_current = self.desired
        self._desired_min = 0.5
        self._desired_max = 2.0

        # Min-kanyar segéd
        self._rescue_r45_ratio = 0.9
        self._rescue_angle = 0.10
        self._rescue_gain = 0.50

        # --- RViz vizualizáció ---
        self.viz_enable = bool(self.declare_parameter('viz_enable', True).value)
        self.viz_frame = self.declare_parameter('viz_frame', 'base_link').get_parameter_value().string_value
        self.viz_history_len = int(self.declare_parameter('viz_history_len', 200).value)
        self.viz_point_size = float(self.declare_parameter('viz_point_size', 0.12).value)
        self.viz_line_width = float(self.declare_parameter('viz_line_width', 0.03).value)
        self.viz_lifetime = float(self.declare_parameter('viz_lifetime', 0.0).value)

        self.goal_hist = deque(maxlen=self.viz_history_len)
        self.viz_pub = self.create_publisher(MarkerArray, 'goal_viz', 1)

        self.viz_base_frame = 'base_link'
        self.tf_buffer = Buffer()
               # noqa: E702
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Állapotok
        self.int_err = 0.0
        self.prev_err = 0.0
        self.prev_t = None
        self._w_filt = 0.0
        self._last_w = 0.0
        self._reversal_until = 0.0
        self.last_switch_t = 0.0

        # goal jump detektálás állapot
        self._last_goal = None
        self._gp_jump_count = 0

        # QoS
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, qos)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.med_err = MedianFilter(self.median_window)
        self.get_logger().info(
            f'teszt2_wall_follower running. side={self.side}, desired={self.desired} m'
        )

    # ---------- angle helper ----------
    def _wrap_angle_to_scan(self, angle_rad: float, scan: LaserScan) -> float:
        amin, amax = scan.angle_min, scan.angle_max
        width = amax - amin
        if width == 0.0:
            return amin
        a = (angle_rad - amin) % (2.0 * math.pi)
        if a > width:
            a -= 2.0 * math.pi
        return amin + a

    def _angle_to_index(self, scan: LaserScan, angle_rad: float) -> int:
        a_wrapped = self._wrap_angle_to_scan(angle_rad, scan)
        inc = scan.angle_increment
        if inc == 0.0:
            return 0
        idx = int(round((a_wrapped - scan.angle_min) / inc))
        return clamp(idx, 0, len(scan.ranges) - 1)

    # ---------- LiDAR segéd ----------
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

    # ---------- Geometria (backup PID-hez) ----------
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

    # ---------- Goalpoint 45° alapján + SAROKTARTALÉK ----------
    def _goalpoint_from_45(self, scan: LaserScan, now_s: float):
        a = math.radians(self.sign * self.follow_angle_deg)
        r = self._get_range_at(scan, a, half_window=1)
        if not math.isfinite(r) or r >= self.range_clip_max:
            return None
        px, py = r * math.cos(a), r * math.sin(a)
        norm = math.hypot(px, py)
        if norm < 1e-6:
            return None
        nx, ny = -px / norm, -py / norm

        # Saroktartalék ideje alatt ideiglenesen nagyobb “desired”
        extra = self.post_switch_expand_add if (now_s < self._expand_until) else 0.0
        desired_eff = self.desired_current + extra

        gx = px + desired_eff * nx + self.lookahead_forward
        gy = py + desired_eff * ny
        return gx, gy

    # ---------- BELSŐ: oldalváltás végrehajtása ----------
    def _do_switch(self, now_s: float, scan: LaserScan, reason: str):
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
        self._w_filt = 0.0
        self._last_w = 0.0

        # dinamikus cél beállítás (új oldal 90°-a) – de SOHA ne legyen kisebb a desired-nél
        d_new = self._avg_range_in_sector(scan, math.radians(self.sign * 90.0), self.side_presence_sector_deg)
        if not math.isfinite(d_new):
            d_new = self.desired
        self.desired_current = clamp(max(self.desired, d_new), self._desired_min, self._desired_max)

        # SAROKTARTALÉK aktiválása
        self._expand_until = now_s + self.post_switch_expand_s

        # jump detektor reset
        self._last_goal = None
        self._gp_jump_count = 0

        self.get_logger().info(
            f'[SWITCH] {old_side} -> {self.side}; reason={reason}; '
            f'desired_cur={self.desired_current:.2f}; expand {self.post_switch_expand_add:.2f}m/{self.post_switch_expand_s:.1f}s'
        )

    # ---------- Oldalváltás (régi – 90° jelenlét alapján) ----------
    def _maybe_switch_side_presence(self, scan: LaserScan, now_s: float):
        if not self.presence_switch_enable:
            return
        side_center = math.radians(self.sign * 90.0)
        d_avg = self._avg_range_in_sector(scan, side_center, self.side_presence_sector_deg)
        r45 = self._get_range_at(scan, math.radians(self.sign * 45.0), 1)
        if not math.isfinite(r45):
            r45 = self.range_clip_max
        need_switch = (d_avg > self.side_presence_max) or (r45 > 0.9 * self.side_presence_max)
        if need_switch and (now_s - self.last_switch_t) > self.side_switch_hold_s:
            self._do_switch(now_s, scan, reason=f'presence d90={d_avg:.2f} r45={r45:.2f}')

    # ---------- RViz publikálás ----------
    def _publish_goal_markers(self, gx, gy, now_s: float):
        p = self._to_viz_frame(gx, gy)
        if p is None:
            return
        gx, gy = p
        if not self.viz_enable:
            return

        self.goal_hist.append((gx, gy))

        ma = MarkerArray()

        m_goal = Marker()
        m_goal.header.frame_id = self.viz_frame
        m_goal.header.stamp = Time().to_msg()
        m_goal.ns = "teszt2_goal"
        m_goal.id = 0
        m_goal.type = Marker.SPHERE
        m_goal.action = Marker.ADD
        m_goal.pose.position.x = float(gx)
        m_goal.pose.position.y = float(gy)
        m_goal.pose.position.z = 0.0
        m_goal.pose.orientation.w = 1.0
        m_goal.scale.x = self.viz_point_size
        m_goal.scale.y = self.viz_point_size
        m_goal.scale.z = self.viz_point_size
        m_goal.color.a = 1.0
        m_goal.color.r = 0.1
        m_goal.color.g = 0.95
        m_goal.color.b = 0.2
        m_goal.lifetime.sec = int(self.viz_lifetime)
        ma.markers.append(m_goal)

        m_line = Marker()
        m_line.header.frame_id = self.viz_frame
        m_line.header.stamp = m_goal.header.stamp
        m_line.ns = "teszt2_goal"
        m_line.id = 1
        m_line.type = Marker.LINE_STRIP
        m_line.action = Marker.ADD
        m_line.scale.x = self.viz_line_width
        m_line.color.a = 0.9
        m_line.color.r = 0.2
        m_line.color.g = 0.6
        m_line.color.b = 1.0
        m_line.lifetime.sec = int(self.viz_lifetime)

        from geometry_msgs.msg import Point
        for (x, y) in self.goal_hist:
            p = Point(); p.x, p.y, p.z = float(x), float(y), 0.0
            m_line.points.append(p)
        ma.markers.append(m_line)

        self.viz_pub.publish(ma)

    def _to_viz_frame(self, x, y):
        if self.viz_frame == self.viz_base_frame:
            return x, y
        try:
            t = self.tf_buffer.lookup_transform(self.viz_frame, self.viz_base_frame, Time())
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            q = t.transform.rotation
            siny_cosp = 2*(q.w*q.z + q.x*q.y)
            cosy_cosp = 1 - 2*(q.y*q.y + q.z*q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            cy = math.cos(yaw); sy = math.sin(yaw)
            X = tx +  cy*x - sy*y
            Y = ty +  sy*x + cy*y
            return X, Y
        except Exception:
            return None

    # ---------- Dinamikus LPF ----------
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

    # ---------- fő callback ----------
    def scan_cb(self, scan: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.prev_t is None else max(1e-3, now - self.prev_t)
        self.prev_t = now

        theta = 0.0
        a = float('nan')
        b = float('nan')
        d_t = float('nan')
        err = 0.0

        # opcionális régi (90°) váltás
        self._maybe_switch_side_presence(scan, now)

        min_fwd = self._min_forward(scan)

        # --- desired_current fokozatos visszahúzás a végső desired felé ---
        if self.desired_relax_tau_s > 1e-3:
            # exp. simítás: y += alpha*(target - y), alpha ~ dt/tau (pontosabban 1-exp(-dt/tau))
            alpha = 1.0 - math.exp(-dt / self.desired_relax_tau_s)
            self.desired_current += alpha * (self.desired - self.desired_current)
            self.desired_current = clamp(self.desired_current, self._desired_min, self._desired_max)

        w_from_goal = None
        gp = None
        gp_raw = None

        if self.use_goalpoint:
            gp_raw = self._goalpoint_from_45(scan, now)

            # --- goal jump alapú oldalváltás ---
            if self.gp_jump_enable and gp_raw is not None:
                if self._last_goal is not None:
                    dx = gp_raw[0] - self._last_goal[0]
                    dy = gp_raw[1] - self._last_goal[1]
                    jump = math.hypot(dx, dy)
                    if jump > self.gp_jump_thresh:
                        self._gp_jump_count += 1
                    else:
                        self._gp_jump_count = 0

                    if (self._gp_jump_count >= self.gp_jump_frames and
                        (now - self.last_switch_t) > self.side_switch_hold_s):
                        self._do_switch(now, scan, reason=f'goal_jump {jump:.2f} m')
                        gp_raw = self._goalpoint_from_45(scan, now)
                        self._gp_jump_count = 0
                self._last_goal = gp_raw

            if gp_raw is not None:
                gp = gp_raw
                gx, gy = gp
                ang_goal = math.atan2(gy, gx)
                w_from_goal = self.k_goal * ang_goal

                # min-kanyar mentőöv
                r45_now = self._get_range_at(scan, math.radians(self.sign * 45.0), 1)
                if not math.isfinite(r45_now):
                    r45_now = self.range_clip_max
                if (r45_now > self._rescue_r45_ratio * self.side_presence_max) and (abs(ang_goal) < self._rescue_angle):
                    w_from_goal = self.sign * self._rescue_gain * self.w_lim

        if w_from_goal is not None:
            w_raw = w_from_goal
            v = self.v_lin
            gamma = 2.0
            turn_ratio = clamp(abs(w_raw) / max(self.w_lim, 1e-6), 0.0, 1.0)
            v = max(self.v_lin_min, self.v_lin * max(0.1, 1.0 - (turn_ratio ** gamma)))
            if min_fwd < self.slow_down_dist:
                v = max(self.v_lin_min, v * 0.35)
            if min_fwd < self.stop_dist:
                v = 0.0
                w = (-self.sign) * 0.8 * self.w_lim
                self._w_filt = w
            else:
                w = self._filter_and_clamp_w(w_raw, now)
        else:
            # backup PID (ha nincs értelmes goalpoint)
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
            v = max(self.v_lin_min, self.v_lin * max(0.1, 1.0 - (turn_ratio ** gamma)))
            if min_fwd < self.slow_down_dist:
                v = max(self.v_lin_min, v * 0.35)
            if min_fwd < self.stop_dist:
                v = 0.0
                w = (-self.sign) * 0.8 * self.w_lim
                self._w_filt = w
            else:
                w = self._filter_and_clamp_w(w_raw, now)

        # parancs
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)

        # RViz marker (ha van friss goalpoint)
        if gp_raw is not None:
            gx, gy = gp_raw
            self._publish_goal_markers(gx, gy, now)


def main():
    rclpy.init()
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
