import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from collections import deque
import cv2
import math 

def build_dist_map_bruteforce(occupancy_grid_msg, max_dist=2.0):
    """
    occupancy_grid_msg: ROS nav_msgs/OccupancyGrid 메시지
    max_dist: 최대 거리 (미터)
    return: 거리맵(dist_map) [m] (numpy array)
    """

    # --- 1. 기본 정보 추출 ---
    width  = occupancy_grid_msg.info.width
    height = occupancy_grid_msg.info.height
    res    = occupancy_grid_msg.info.resolution
    data   = np.array(occupancy_grid_msg.data, dtype=np.int8).reshape(height, width)

    # --- 2. 초기화 ---
    dist_sq = np.full((height, width), max_dist * max_dist, dtype=np.float32)
    l = int(math.ceil(max_dist / res))

    # --- 3. 각 셀 전체 탐색 (bruteforce) ---
    for j in range(width):          # ← C++의 outer loop: j = x index
        for k in range(height):     # ← C++ inner loop: k = y index
            if data[k, j] > 0:      # isObstacle(j, k)
                # 장애물 주변에 거리 채워 넣기
                for x in range(max(0, j - l), min(width, j + l + 1)):
                    for y in range(max(0, k - l), min(height, k + l + 1)):
                        dx = (x - j)
                        dy = (y - k)
                        d2 = (dx * dx + dy * dy) * res * res
                        if d2 < dist_sq[y, x]:
                            dist_sq[y, x] = d2

    # --- 4. sqrt로 실제 거리로 변환 ---
    dist_map = np.sqrt(dist_sq)

    # 거리 제한
    dist_map[dist_map > max_dist] = max_dist

    return dist_map

def show_dist_map_cv(dist_map, max_dist=2.0):
    """거리맵을 OpenCV 창에 컬러 heatmap으로 표시"""
    norm = np.clip(dist_map / max_dist, 0, 1)
    img_gray = norm * 255.0  # 가까울수록 어둡게 (파란색이 안전,빨간색이 위험)
    img_gray = img_gray.astype(np.uint8)
        
    color_img = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
    color_img = cv2.rotate(color_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #color_img = cv2.flip(color_img, 0)

    cv2.imshow("Distance Map (OpenCV)", color_img)
    cv2.waitKey(1)


class DistMapVisualizer(Node):
    def __init__(self):
        super().__init__('distmap_visualizer')

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/bev/occupancy_grid',
            self.map_callback,
            10
        )
        self.get_logger().info("✅ Subscribed to /bev/occupancy_grid (OpenCV visualization enabled)")
        self.frame_count = 0

    def map_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            # 너무 자주 계산하면 CPU 과부하 → 5프레임에 1회만 표시
            return

        self.get_logger().info(f"🗺️  Frame {self.frame_count}: map size {msg.info.width}x{msg.info.height}")
        dist_map = build_dist_map_bruteforce(msg, max_dist=2.0)
        show_dist_map_cv(dist_map, max_dist=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = DistMapVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
