#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import time
from collections import deque

from distmap_cuda_def import build_dist_map_bfs_cuda,distmap_to_occupancygrid



class DistMapPublisher(Node):
    def __init__(self):
        super().__init__('distmap_publisher')

        self.subscriber = self.create_subscription(
            OccupancyGrid,
            '/bev/occupancy_grid',
            self.map_callback,
            10
        )
        self.publisher = self.create_publisher(
            OccupancyGrid,
            '/dist_map',
            10
        )

        self.get_logger().info("✅ Subscribed to /bev/occupancy_grid")
        self.get_logger().info("✅ Publishing GPU distance map to /dist_map")

        # 최근 프레임 시간 저장용 (최근 30프레임)
        self.frame_times = deque(maxlen=30)

    def map_callback(self, msg: OccupancyGrid):
        start = time.time()

        # 거리맵 계산 (GPU)
        dist_map = build_dist_map_bfs_cuda(msg, max_dist=2.0)

        # OccupancyGrid로 변환 및 퍼블리시
        dist_msg = distmap_to_occupancygrid(dist_map, msg, max_dist=2.0)
        self.publisher.publish(dist_msg)

        self.get_logger().info("Published /dist_map")

def main(args=None):
    rclpy.init(args=args)
    node = DistMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
