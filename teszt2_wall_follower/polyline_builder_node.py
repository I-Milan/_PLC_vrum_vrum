
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PolylineBuilder(Node):
    def __init__(self):
        super().__init__("polyline_builder")

        # ---- paraméterek ----
        self.frame_id = (
            self.declare_parameter("frame_id", "base_link")
            .get_parameter_value()
            .string_value
        )
        self.valid_min = float(self.declare_parameter("valid_min", 0.03).value)
        self.range_clip_max = float(self.declare_parameter("range_clip_max", 5.0).value)

        # az elülső szög, amit KIDOBUNK (mindkét oldalon!)
        self.front_cut_deg = float(self.declare_parameter("front_cut_deg", 25.0).value)
        self.front_cut_rad = math.radians(self.front_cut_deg)

        # csak a legközelebbi fal környékét hagyjuk meg:
        # r <= r_min + keep_radius_delta
        self.keep_radius_delta = float(
            self.declare_parameter("keep_radius_delta", 0.8).value
        )

        # pont-összekötés limitjei 
        self.max_seg_len = float(self.declare_parameter("max_seg_len", 0.6).value)
        self.max_angle_step_rad = math.radians(
            float(self.declare_parameter("max_angle_step_deg", 4.0).value)
        )

        # viz
        self.viz_enable = bool(self.declare_parameter("viz_enable", True).value)
        self.viz_lifetime = float(self.declare_parameter("viz_lifetime", 0.0).value)

        # QoS + I/O
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "scan", self.scan_cb, qos
        )
        self.pub_left = self.create_publisher(Path, "polyline_left", 1)
        self.pub_right = self.create_publisher(Path, "polyline_right", 1)
        self.viz_pub = self.create_publisher(MarkerArray, "polyline_viz", 1)

        self.get_logger().info(
            f"[polyline_builder] running. scan: /scan, frame: {self.frame_id}"
        )

    # ------------ path publikálás ----------------
    def _path_from_points(self, pts):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.frame_id
        for x, y in pts:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    # ------------ vizualizáció -------------------
    def _publish_viz(self, left_pts, right_pts):
        if not self.viz_enable:
            return

        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def make_line(id_, pts, r, g, b):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "polyline"
            m.id = id_
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.04
            m.color.a = 1.0
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.lifetime.sec = int(self.viz_lifetime)
            from geometry_msgs.msg import Point

            for x, y in pts:
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                m.points.append(p)
            return m

        ma.markers.append(make_line(0, left_pts, 0.0, 1.0, 0.0))
        ma.markers.append(make_line(1, right_pts, 1.0, 0.0, 0.0))

        self.viz_pub.publish(ma)

    # ------------ fő callback ---------------------------
    def scan_cb(self, scan: LaserScan):
        # 1) pontok szétválogatása bal/jobb oldalra, front kivágással
        pts_left = []
        pts_right = []

        angle = scan.angle_min
        inc = scan.angle_increment

        for r in scan.ranges:
            if not math.isfinite(r) or r < self.valid_min:
                angle += inc
                continue
            r = min(r, self.range_clip_max)

            # front zóna eldobása MINDKÉT oldalon
            if abs(angle) < self.front_cut_rad:
                angle += inc
                continue

            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if y >= 0.0:
                pts_left.append((x, y, angle, r))
            else:
                pts_right.append((x, y, angle, r))

            angle += inc

        # ha valamelyik oldalon nincs pont, ne essünk szét
        def filter_by_radius(pts):
            if not pts:
                return []
            r_min = min(p[3] for p in pts)
            limit = r_min + self.keep_radius_delta
            return [p for p in pts if p[3] <= limit]

        pts_left = filter_by_radius(pts_left)
        pts_right = filter_by_radius(pts_right)

        # rendezés szög szerint (szép vonal miatt)
        pts_left.sort(key=lambda p: p[2])
        pts_right.sort(key=lambda p: p[2])

        # lokális lánc: túl nagy szög / távolság ugrásnál ne kössük össze
        def build_chain(pts):
            if not pts:
                return []
            chain = [pts[0][:2]]  # csak (x,y)
            prev = pts[0]
            for p in pts[1:]:
                dx = p[0] - prev[0]
                dy = p[1] - prev[1]
                d = math.hypot(dx, dy)
                d_alpha = abs(p[2] - prev[2])
                if d <= self.max_seg_len and d_alpha <= self.max_angle_step_rad:
                    chain.append(p[:2])
                    prev = p
                else:
                    # szakadjon meg a vonal – csak a közelebbi szegmens érdekel,
                    # ezért itt egyszerűen új "kezdetet" veszünk, de nem kötjük össze
                    chain.append(p[:2])
                    prev = p
            return chain

        chain_left = build_chain(pts_left)
        chain_right = build_chain(pts_right)

        # 2) Path üzenetek publikálása
        self.pub_left.publish(self._path_from_points(chain_left))
        self.pub_right.publish(self._path_from_points(chain_right))

        # 3) RViz marker
        self._publish_viz(chain_left, chain_right)


def main():
    rclpy.init()
    node = PolylineBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
