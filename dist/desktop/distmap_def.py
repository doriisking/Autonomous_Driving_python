# distmap.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from scipy.ndimage import distance_transform_edt

import numpy as np
import math
from collections import deque

def build_dist_map_bfs(occupancy_grid_msg, max_dist=5.0):
    """
    occupancy_grid_msg : ROS OccupancyGrid 메시지
    max_dist           : 거리 제한 [m]
    반환값             : 각 셀에서 가장 가까운 장애물까지 거리 [m]
    """
    width  = occupancy_grid_msg.info.width
    height = occupancy_grid_msg.info.height
    res    = occupancy_grid_msg.info.resolution
    data   = np.array(occupancy_grid_msg.data, dtype=np.int8).reshape(height, width)

    # 장애물 마스크 생성 (0 = free, 100 = obstacle)
    obstacle_mask = (data > 0)
    dist_map = np.full((height, width), np.inf, dtype=np.float32)

    # BFS 큐 초기화 (장애물 셀들을 시작점으로)
    q = deque()
    for y in range(height):
        for x in range(width):
            if obstacle_mask[y, x]:
                dist_map[y, x] = 0.0
                q.append((x, y))

    # 8방향 이동 정의
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # BFS 거리 전파
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

    dist_map[dist_map > max_dist] = max_dist
    return dist_map


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

def get_dist_from_map(x_world, y_world, dist_map, origin_x, origin_y, resolution):
    """
    실제 좌표(x,y) [m]에서 장애물까지의 거리 [m] 반환
    - x_world, y_world : 실제 좌표 (OccupancyGrid frame)
    - dist_map         : build_dist_map_bfs()로 생성된 거리맵
    - origin_x, origin_y : OccupancyGrid의 origin (좌하단 좌표)
    - resolution       : grid cell 크기 [m]
    """
    height, width = dist_map.shape
    # 좌표 → 그리드 인덱스
    ix = int(round((x_world - origin_x) / resolution))
    iy = int(round((y_world - origin_y) / resolution))

    # 경계 보정
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)

    # 거리 리턴
    return float(dist_map[iy, ix])

def distmap_to_occupancygrid(dist_map, template_msg, max_dist=5.0):
    """
    dist_map: numpy array (height x width), 거리[m]
    template_msg: 원본 OccupancyGrid (geometry/metadata 복사용)
    return: OccupancyGrid (data scaled to 0~100)
    """
    msg = OccupancyGrid()
    msg.header = Header()
    msg.header.frame_id = template_msg.header.frame_id
    msg.info = template_msg.info

    # 0~max_dist → 0~100 스케일 (float→int8)
    scaled = np.clip(dist_map / max_dist * 100, 0, 100)
    msg.data = scaled.astype(np.int8).flatten().tolist()
    return msg