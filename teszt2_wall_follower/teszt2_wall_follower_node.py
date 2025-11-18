
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class MedianFilter:
    def __init__(self, window):
        self.buf = deque(maxlen=window)

    def push(self, x):
        self.buf.append(x)
        if not self.buf:
            return x
        s = sorted(self.buf)
        return s[len(s)//2]


class WallFollower(Node):
    def __init__(self):
        super().__init__('teszt2_wall_follower')

        # --- Alap paramok ---
        self.side = self.declare_parameter('side', 'left').get_parameter_value().string_value
        assert self.side in ('left', 'right'), 'side param must be "left" or "right"'
        self.sign = 1.0 if self.side == 'left' else -1.0

        self.desired = float(self.declare_parameter('desired_distance', 1.5).value)
        self.desired_current = self.desired
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

        # Egyszerű LPF a kormányra
        self.steer_lpf_alpha = float(self.declare_parameter('steer_lpf_alpha', 0.5).value)

        # Goalpoint mód (45°-os követés)
        self.follow_angle_deg = float(self.declare_parameter('follow_angle_deg', 45.0).value)
        self.lookahead_forward = float(self.declare_parameter('lookahead_forward', 1.00).value)
        self.k_goal = float(self.declare_parameter('k_goal', 1.0).value)
        self.use_goalpoint = bool(self.declare_parameter('use_goalpoint', True).value)

        # Polyline használat
        self.use_polyline = bool(self.declare_parameter('use_polyline', True).value)
        self.min_poly_points = int(self.declare_parameter('min_poly_points', 6).value)
        # outlier limit a goalpontra
        self.max_goal_jump = float(self.declare_parameter('max_goal_jump', 2.0).value)

        # Oldalváltás goal-jump alapján
        self.gp_jump_enable = bool(self.declare_parameter('gp_jump_enable', True).value)
        self.gp_jump_thresh = float(self.declare_parameter('gp_jump_thresh', 1.8).value)
        self.gp_jump_frames = int(self.declare_parameter('gp_jump_frames', 2).value)
        self.side_switch_hold_s = float(self.declare_parameter('side_switch_hold_s', 1.0).value)

        # Vizualizáció (goal pont + trail)
        self.viz_enable = bool(self.declare_parameter('viz_enable', True).value)
        self.viz_frame = self.declare_parameter('viz_frame', 'base_link').get_parameter_value().string_value
        self.viz_history_len = int(self.declare_parameter('viz_history_len', 200).value)
        self.viz_point_size = float(self.declare_parameter('viz_point_size', 0.12).value)
        self.viz_line_width = float(self.declare_parameter('viz_line_width', 0.03).value)
        self.viz_lifetime = float(self.declare_parameter('viz_lifetime', 0.0).value)

        # Származtatott szögek az aktuális oldalhoz
        self.angle_a = math.radians(self.sign * self.angle_a_deg)  # 90°
        self.angle_b = math.radians(self.sign * self.angle_b_deg)  # 45°

        # Állapotok
        self.int_err = 0.0
        self.prev_err = 0.0
        self.prev_t = None
        self._w_filt = 0.0
        self._last_w = 0.0
        self._last_goal = None
        self._gp_jump_count = 0
        self.last_switch_t = 0.0

        # TF a vizhez
        self.viz_base_frame = 'base_link'
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_hist = deque(maxlen=self.viz_history_len)
        self.viz_pub = self.create_publisher(MarkerArray, 'goal_viz', 1)

        # QoS + I/O
        qos_scan = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, qos_scan)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Polyline Path-ok (külön polyline_builder node publikálja)
        self.poly_left = []
        self.poly_right = []
        self.poly_frame = "base_link"
        self.create_subscription(Path, 'polyline_left',  self._poly_cb_left,  10)
        self.create_subscription(Path, 'polyline_right', self._poly_cb_right, 10)

        self.med_err = MedianFilter(self.median_window)

        self.get_logger().info(
            f'teszt2_wall_follower running. side={self.side}, desired={self.desired:.2f} m'
        )

    # ------------ Polyline callbackek ------------
    def _poly_cb_left(self, msg: Path):
        self.poly_left = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self.poly_frame = msg.header.frame_id or "base_link"

    def _poly_cb_right(self, msg: Path):
        self.poly_right = [(ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self.poly_frame = msg.header.frame_id or "base_link"

    def _poly_active(self):
        pl = self.poly_left if self.side == 'left' else self.poly_right
        return self.use_polyline and (len(pl) >= self.min_poly_points)

    # ------------ LiDAR helpers ------------
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

    def _get_range_at_scan(self, scan: LaserScan, angle_rad: float, half_window: int = 1) -> float:
        i_center = self._angle_to_index(scan, angle_rad)
        lo = clamp(i_center - half_window, 0, len(scan.ranges) - 1)
        hi = clamp(i_center + half_window, 0, len(scan.ranges) - 1)
        vals = []
        for i in range(lo, hi + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and r >= self.valid_min:
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
        lo, hi = (min(idx_lo, idx_hi), max(idx_lo, idx_hi))
        mins = self.range_clip_max
        for i in range(lo, hi + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and r >= self.valid_min:
                mins = min(mins, min(r, self.range_clip_max))
        return mins

    # ------------ Backup PID geometria (nyers LiDAR) ------------
    def _compute_error(self, scan: LaserScan):
        beta = abs(self.angle_a - self.angle_b)
        a = self._get_range_at_scan(scan, self.angle_a, half_window=1)
        b = self._get_range_at_scan(scan, self.angle_b, half_window=1)

        a = min(a, self.range_clip_max)
        b = min(b, self.range_clip_max)

        theta = math.atan2(a * math.cos(beta) - b, a * math.sin(beta))
        d_t = b * math.cos(theta) - self.L * math.sin(theta)
        err = self.desired - d_t
        return err, theta, a, b, d_t

    # ------------ Raycast polyline mentén ------------
    def _raycast_polyline(self, angle_rad: float):
        poly = self.poly_left if self.side == 'left' else self.poly_right
        if len(poly) < self.min_poly_points:
            return None

        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        best_t = None
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            x2, y2 = poly[i + 1]
            sx = x2 - x1
            sy = y2 - y1

            # ray [0,0]+t[dx,dy], seg [x1,y1]+u[sx,sy]
            det = (-dx * sy + dy * sx)
            if abs(det) < 1e-9:
                continue

            u = (-dy * x1 + dx * y1) / det

            if abs(dx) > abs(dy):
                t = (x1 + u * sx) / (dx if abs(dx) > 1e-9 else 1e-9)
            else:
                t = (y1 + u * sy) / (dy if abs(dy) > 1e-9 else 1e-9)

            if 0.0 <= u <= 1.0 and t is not None and t >= self.valid_min:
                if best_t is None or t < best_t:
                    best_t = t

        if best_t is not None and math.isfinite(best_t):
            return min(best_t, self.range_clip_max)
        return None

    # ------------ Range follow-angle-nál (Polyline elsődleges) ------------
    def _get_range_follow(self, scan: LaserScan, angle_rad: float) -> float:
        r = None
        if self._poly_active():
            r = self._raycast_polyline(angle_rad)
        if r is None:
            # fallback: LiDAR
            r = self._get_range_at_scan(scan, angle_rad, half_window=1)
        return r

    # ------------ Goalpoint follow-angle alapján ------------
    def _goalpoint_from_follow(self, scan: LaserScan):
        a = math.radians(self.sign * self.follow_angle_deg)
        r = self._get_range_follow(scan, a)
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

    # ------------ RViz vizualizáció ------------
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
            cy = math.cos(yaw)
            sy = math.sin(yaw)
            X = tx + cy * x - sy * y
            Y = ty + sy * x + cy * y
            return X, Y
        except Exception:
            return None

    def _publish_goal_markers(self, gx, gy, now_s: float):
        if not self.viz_enable:
            return
        p = self._to_viz_frame(gx, gy)
        if p is None:
            return
        gx, gy = p

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
        m_goal.pose.orientation.w = 1.0
        m_goal.scale.x = self.viz_point_size
        m_goal.scale.y = self.viz_point_size
        m_goal.scale.z = self.viz_point_size
        m_goal.color.a = 1.0
        m_goal.color.r = 0.1
        m_goal.color.g = 0.95
        m_goal.color.b = 0.2
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

        from geometry_msgs.msg import Point
        m_line.points = [Point(x=float(x), y=float(y), z=0.0) for (x, y) in self.goal_hist]
        ma.markers.append(m_line)

        self.viz_pub.publish(ma)

    # ------------ LPF + clamp ------------
    def _filter_and_clamp_w(self, w):
        a = clamp(self.steer_lpf_alpha, 0.0, 1.0)
        self._w_filt = (1.0 - a) * self._w_filt + a * w
        return clamp(self._w_filt, -self.w_lim, self.w_lim)

    # ------------ fő callback ------------
    def scan_cb(self, scan: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.prev_t is None else max(1e-3, now - self.prev_t)
        self.prev_t = now

        min_fwd = self._min_forward(scan)

        gp_raw = None
        if self.use_goalpoint:
            gp_raw = self._goalpoint_from_follow(scan)

            # goal-jump alapú oldalváltás (ha bekapcsolva)
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
                        old = self.side
                        self.sign *= -1.0
                        self.side = 'left' if self.sign > 0 else 'right'
                        self.last_switch_t = now
                        self.angle_a = math.radians(self.sign * self.angle_a_deg)
                        self.angle_b = math.radians(self.sign * self.angle_b_deg)
                        self._w_filt = 0.0
                        self._last_goal = None
                        self._gp_jump_count = 0
                        self.get_logger().info(f'[SWITCH] {old} -> {self.side} (goal jump {jump:.2f} m)')
                        gp_raw = self._goalpoint_from_follow(scan)
                self._last_goal = gp_raw

        # --------- Goalpoint ág ---------
        if gp_raw is not None:
            gp = gp_raw
            if self._last_goal is not None:
                dx = gp_raw[0] - self._last_goal[0]
                dy = gp_raw[1] - self._last_goal[1]
                if math.hypot(dx, dy) > self.max_goal_jump:
                    gp = self._last_goal
            self._last_goal = gp

            gx, gy = gp
            ang_goal = math.atan2(gy, gx)
            w_raw = self.k_goal * ang_goal

            turn_ratio = clamp(abs(w_raw) / max(self.w_lim, 1e-6), 0.0, 1.0)
            v = max(self.v_lin_min, self.v_lin * max(0.1, 1.0 - (turn_ratio ** 2.0)))

            if min_fwd < self.slow_down_dist:
                v = max(self.v_lin_min, v * 0.35)
            if min_fwd < self.stop_dist:
                v = 0.0
                w = (-self.sign) * 0.8 * self.w_lim
                self._w_filt = w
            else:
                w = self._filter_and_clamp_w(w_raw)

            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = w
            self.cmd_pub.publish(cmd)

            self._publish_goal_markers(gx, gy, now)

        else:
            # --------- Backup PID (nyers LiDAR) ---------
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

            turn_ratio = clamp(abs(w_raw) / max(self.w_lim, 1e-6), 0.0, 1.0)
            v = max(self.v_lin_min, self.v_lin * max(0.1, 1.0 - (turn_ratio ** 2.0)))

            if min_fwd < self.slow_down_dist:
                v = max(self.v_lin_min, v * 0.35)
            if min_fwd < self.stop_dist:
                v = 0.0
                w = (-self.sign) * 0.8 * self.w_lim
                self._w_filt = w
            else:
                w = self._filter_and_clamp_w(w_raw)

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


if __name__ == "__main__":
    main()
