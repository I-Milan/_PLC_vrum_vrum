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


class AdvancedWallFollower(Node):
    def __init__(self):
        super().__init__('advanced_wall_follower')

        # --- Parameters ---
        self.desired_distance = float(self.declare_parameter('desired_distance', 0.2).value)  # 20 cm
        
        # PID parameters for 90° wall following
        self.kp = float(self.declare_parameter('kp', 3.0).value)
        self.ki = float(self.declare_parameter('ki', 0.0).value)
        self.kd = float(self.declare_parameter('kd', 0.4).value)
        
        # Speed parameters
        self.v_lin = float(self.declare_parameter('linear_speed', 0.8).value)
        self.v_lin_min = float(self.declare_parameter('linear_speed_min', 0.2).value)
        self.w_lim = float(self.declare_parameter('angular_speed_limit', 3.0).value)
        
        # Safety distances
        self.slow_down_dist = float(self.declare_parameter('slow_down_distance', 0.6).value)
        self.stop_dist = float(self.declare_parameter('stop_distance', 0.3).value)
        
        # Filter parameters
        self.median_window = int(self.declare_parameter('median_window', 5).value)
        self.range_clip_max = float(self.declare_parameter('range_clip_max', 3.0).value)
        self.valid_min = float(self.declare_parameter('valid_min', 0.03).value)
        
        # Turn detection thresholds
        self.turn_threshold = float(self.declare_parameter('turn_threshold', 1.5).value)
        self.wall_disappear_threshold = float(self.declare_parameter('wall_disappear_threshold', 2.0).value)
        
        # Maximum approach angle (degrees)
        self.max_approach_angle = math.radians(float(self.declare_parameter('max_approach_angle', 15.0).value))
        
        # Angles for triple sensing (degrees)
        self.angle_90 = math.radians(90)
        self.angle_45 = math.radians(45) 
        self.angle_25 = math.radians(25)
        
        # State variables
        self.current_turn_side = None  # 'left', 'right', or None
        self.turn_initiated = False
        
        # PID variables
        self.int_err = 0.0
        self.prev_err = 0.0
        self.prev_t = None
        
        # Filters
        self.med_err = MedianFilter(self.median_window)
        self.med_90_left = MedianFilter(self.median_window)
        self.med_90_right = MedianFilter(self.median_window)
        self.med_45_left = MedianFilter(self.median_window)
        self.med_45_right = MedianFilter(self.median_window)
        self.med_25_left = MedianFilter(self.median_window)
        self.med_25_right = MedianFilter(self.median_window)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb, qos)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info('Advanced wall follower started - PARALLEL APPROACH MODE')

    def _angle_to_index(self, scan: LaserScan, angle_rad: float) -> int:
        inc = scan.angle_increment
        if inc == 0.0:
            return 0
        idx = int(round((angle_rad - scan.angle_min) / inc))
        return clamp(idx, 0, len(scan.ranges) - 1)

    def _get_range_at(self, scan: LaserScan, angle_rad: float, half_window: int = 1) -> float:
        i_center = self._angle_to_index(scan, angle_rad)
        lo = clamp(i_center - half_window, 0, len(scan.ranges) - 1)
        hi = clamp(i_center + half_window, 0, len(scan.ranges) - 1)
        vals = []
        for i in range(lo, hi + 1):
            r = scan.ranges[i]
            if math.isfinite(r) and r >= self.valid_min:
                vals.append(min(r, self.range_clip_max))
        if not vals:
            return self.range_clip_max
        return sum(vals) / len(vals)

    def _get_all_ranges(self, scan: LaserScan):
        # Get ranges for all critical angles on both sides
        ranges = {
            'left_90': self.med_90_left.push(self._get_range_at(scan, self.angle_90)),
            'right_90': self.med_90_right.push(self._get_range_at(scan, -self.angle_90)),
            'left_45': self.med_45_left.push(self._get_range_at(scan, self.angle_45)),
            'right_45': self.med_45_right.push(self._get_range_at(scan, -self.angle_45)),
            'left_25': self.med_25_left.push(self._get_range_at(scan, self.angle_25)),
            'right_25': self.med_25_right.push(self._get_range_at(scan, -self.angle_25))
        }
        return ranges

    def _detect_turn_intent(self, ranges):
        # Detect which side is opening up (25° indicates upcoming turn)
        left_25_diff = ranges['left_25'] - ranges['left_45']
        right_25_diff = ranges['right_25'] - ranges['right_45']
        
        # If one side is significantly opening up more than the other
        if left_25_diff > self.turn_threshold and left_25_diff > right_25_diff:
            return 'right'  # Turn right towards left wall
        elif right_25_diff > self.turn_threshold and right_25_diff > left_25_diff:
            return 'left'   # Turn left towards right wall
        else:
            return None

    def _detect_turn_start(self, ranges, turn_side):
        # Detect when wall disappears at 45° (turn actually starts)
        if turn_side == 'right':
            return ranges['left_45'] >= self.wall_disappear_threshold
        elif turn_side == 'left':
            return ranges['right_45'] >= self.wall_disappear_threshold
        return False

    def _compute_pid_output(self, ranges, dt, turn_side):
        # Use 90° reading for PID control on the opposite wall during turns
        if turn_side == 'right':
            # Follow left wall during right turn
            error = self.desired_distance - ranges['left_90']
            # Párhuzamos közeledés - korlátozzuk a maximális szöget
            max_error_for_angle = math.tan(self.max_approach_angle) * ranges['left_90']
            error = clamp(error, -max_error_for_angle, max_error_for_angle)
            
        elif turn_side == 'left':
            # Follow right wall during left turn  
            error = self.desired_distance - ranges['right_90']
            # Párhuzamos közeledés - korlátozzuk a maximális szöget
            max_error_for_angle = math.tan(self.max_approach_angle) * ranges['right_90']
            error = clamp(error, -max_error_for_angle, max_error_for_angle)
        else:
            # Normal straight - follow closer wall
            left_error = self.desired_distance - ranges['left_90']
            right_error = self.desired_distance - ranges['right_90']
            error = left_error if abs(left_error) < abs(right_error) else right_error
        
        error = self.med_err.push(error)
        
        # PID calculation
        self.int_err += error * dt
        i_max = 1.0
        self.int_err = clamp(self.int_err, -i_max, i_max)
        
        der = (error - self.prev_err) / dt if dt > 0.0 else 0.0
        self.prev_err = error
        
        w = self.kp * error + self.ki * self.int_err + self.kd * der
        
        # Reset integral if error crosses zero
        if error * self.prev_err < 0.0:
            self.int_err *= 0.5
            
        return w, error

    def scan_cb(self, scan: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.0 if self.prev_t is None else max(1e-3, now - self.prev_t)
        self.prev_t = now

        # Get all range measurements
        ranges = self._get_all_ranges(scan)
        
        # Step 1: Detect turn intent using 25° sensors
        if not self.turn_initiated:
            self.current_turn_side = self._detect_turn_intent(ranges)
        
        # Step 2: Check if turn should start (45° wall disappears)
        if self.current_turn_side and not self.turn_initiated:
            if self._detect_turn_start(ranges, self.current_turn_side):
                self.turn_initiated = True
                self.get_logger().info(f'Starting {self.current_turn_side} turn - Parallel approach active')
        
        # Step 3: Compute control outputs
        w_pid, error = self._compute_pid_output(ranges, dt, self.current_turn_side)
        
        # Step 4: Determine final angular velocity
        if self.turn_initiated and self.current_turn_side:
            # Full turn during cornering
            if self.current_turn_side == 'right':
                w = -self.w_lim * 1.2  # Turn right
            else:  # left turn
                w = self.w_lim * 1.2   # Turn left
        else:
            # Normal PID control
            w = clamp(w_pid, -self.w_lim, self.w_lim)
        
        # Step 5: Determine linear velocity
        v = self.v_lin
        
        # Forward looking for obstacles
        forward_min = min(ranges['left_45'], ranges['right_45'], 
                         ranges['left_25'], ranges['right_25'])
        
        # Only slow down when really close
        if forward_min < self.slow_down_dist:
            slow_factor = max(0.6, forward_min / self.slow_down_dist)
            v = max(self.v_lin_min, v * slow_factor)
        
        if forward_min < self.stop_dist:
            v = 0.0
            # Reset turn state if stuck
            self.turn_initiated = False
            self.current_turn_side = None
        
        # Reset turn state when turn is complete
        if self.turn_initiated:
            turn_complete = False
            if self.current_turn_side == 'right' and ranges['left_45'] < self.wall_disappear_threshold:
                turn_complete = True
            elif self.current_turn_side == 'left' and ranges['right_45'] < self.wall_disappear_threshold:
                turn_complete = True
                
            if turn_complete:
                self.turn_initiated = False
                self.current_turn_side = None
                self.get_logger().info(f'{self.current_turn_side} turn completed')
        
        # Deadband for small errors
        if abs(error) < 0.005 and not self.turn_initiated:
            w = 0.0
        
        # Publish command
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)
        
        # Debug info
        self.get_logger().debug(f'V: {v:.2f}, W: {w:.2f}, Turn: {self.current_turn_side}, Error: {error:.3f}')

    def destroy_node(self):
        # Stop the robot when shutting down
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        super().destroy_node()


def main():
    rclpy.init()
    node = AdvancedWallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
