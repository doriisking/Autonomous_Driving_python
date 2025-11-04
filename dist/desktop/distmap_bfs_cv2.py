import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np
from collections import deque
import cv2


def build_dist_map_bfs(occupancy_grid_msg, max_dist=2.0):
    width  = occupancy_grid_msg.info.width
    height = occupancy_grid_msg.info.height
    res    = occupancy_grid_msg.info.resolution
    data   = np.array(occupancy_grid_msg.data, dtype=np.int8).reshape(height, width)

    # 장애물 = True (값 > 0)
    obstacle_mask = (data > 0)
    dist_map = np.full((height, width), np.inf, dtype=np.float32)

    q = deque()
    for y in range(height):
        for x in range(width):
            if obstacle_mask[y, x]:
                dist_map[y, x] = 0.0
                q.append((x, y))

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while q:
        x, y = q.popleft()
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                step = res * (np.sqrt(2) if dx != 0 and dy != 0 else 1.0)
                new_dist = dist_map[y, x] + step
                if new_dist < dist_map[ny, nx] and new_dist <= max_dist:
                    dist_map[ny, nx] = new_dist
                    q.append((nx, ny))

    dist_map[np.isinf(dist_map)] = max_dist
    dist_map[dist_map > max_dist] = max_dist
    return dist_map


def show_dist_map_cv(dist_map, max_dist=2.0):
    """거리맵을 OpenCV 창에 컬러 heatmap으로 표시"""
    norm = np.clip(dist_map / max_dist, 0, 1)
    img_gray = norm * 255.0  
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

        self.get_logger().info(f"Frame {self.frame_count}: map size {msg.info.width}x{msg.info.height}")
        dist_map = build_dist_map_bfs(msg, max_dist=2.0)
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
