import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from distmap import build_dist_map 

class OccupancyGridSubscriber(Node):
    def __init__(self):
        super().__init__('occupancy_grid_sub')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/map',                     # 토픽 이름 (예: /map)
            self.listener_callback,
            10                          # 큐 사이즈
        )
        self.subscription  # prevent unused variable warning

    def listener_calslback(self, msg: OccupancyGrid):
        # 맵 기본 정보 출력
        self.get_logger().info(f"Resolution: {msg.info.resolution}")
        self.get_logger().info(f"Width: {msg.info.width}, Height: {msg.info.height}")
        self.get_logger().info(f"Origin: ({msg.info.origin.position.x}, {msg.info.origin.position.y})")

        # 셀 데이터 접근
        width = msg.info.width
        height = msg.info.height
        data = msg.data   # 1차원 배열 (길이 = width * height)
        
        ### dist map 계산 

        dist_map = build_dist_map(msg, max_dist=2.0)

        # 예시: (10, 20) 셀의 점유 상태 출력
        x, y = 10, 20
        idx = x + y * width
        cell_value = data[idx]
        self.get_logger().info(f"Cell[{x}, {y}] = {cell_value}")

        import matplotlib.pyplot as plt
        plt.imshow(dist_map, cmap='jet')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
