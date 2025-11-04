import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from scipy.ndimage import distance_transform_edt

def build_dist_map_bfs_cuda(occupancy_grid_msg, max_dist=5.0):
    """
    CUDA 병렬화된 BFS 거리맵 생성
    occupancy_grid_msg : ROS OccupancyGrid 메시지
    max_dist           : 거리 제한 [m]
    """
    width  = occupancy_grid_msg.info.width
    height = occupancy_grid_msg.info.height
    res    = occupancy_grid_msg.info.resolution
    data   = np.array(occupancy_grid_msg.data, dtype=np.int8).reshape(height, width)

    # 장애물 마스크 생성 (1 = obstacle, 0 = free)
    grid_np = (data > 0).astype(np.float32)

    # 거리맵 초기화
    dist_map = np.where(grid_np > 0, 0.0, np.inf).astype(np.float32)

    # CUDA 커널 코드 (wave propagation)
    kernel_code = """
    __global__ void update_distance(
        float *dist, const float *obstacle,
        int width, int height, float res, float max_dist, int *changed)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height)
            return;

        int idx = y * width + x;
        if (obstacle[idx] > 0.5f) return;

        float min_dist = dist[idx];
        float dirs[8][2] = {
            {-1,0},{1,0},{0,-1},{0,1},
            {-1,-1},{-1,1},{1,-1},{1,1}
        };

        for (int i = 0; i < 8; i++) {
            int nx = x + (int)dirs[i][0];
            int ny = y + (int)dirs[i][1];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nidx = ny * width + nx;
                float step = res * ((dirs[i][0] != 0 && dirs[i][1] != 0) ? 1.4142f : 1.0f);
                float new_dist = dist[nidx] + step;
                if (new_dist < min_dist && new_dist <= max_dist) {
                    min_dist = new_dist;
                }
            }
        }

        if (min_dist + 1e-6f < dist[idx]) {
            dist[idx] = min_dist;
            *changed = 1; // BFS 전파 계속
        }
    }
    """

    # CUDA 컴파일
    mod = SourceModule(kernel_code)
    kernel = mod.get_function("update_distance")

    # GPU 메모리 할당
    grid_gpu = cuda.mem_alloc(grid_np.nbytes)
    dist_gpu = cuda.mem_alloc(dist_map.nbytes)
    changed_gpu = cuda.mem_alloc(np.int32().nbytes)

    cuda.memcpy_htod(grid_gpu, grid_np)
    cuda.memcpy_htod(dist_gpu, dist_map)

    block = (16, 16, 1)
    grid_dim = ((width + 15) // 16, (height + 15) // 16)

    # wavefront propagation 반복
    iteration = 0
    while True:
        # ✅ 수정: NumPy 배열로 만들어서 mutable buffer 보장
        changed = np.zeros(1, dtype=np.int32)
        cuda.memcpy_htod(changed_gpu, changed)

        kernel(
            dist_gpu, grid_gpu,
            np.int32(width), np.int32(height),
            np.float32(res), np.float32(max_dist),
            changed_gpu,
            block=block, grid=grid_dim
        )

        cuda.memcpy_dtoh(changed, changed_gpu)
        iteration += 1

        # 변경 없으면 종료
        if changed[0] == 0 or iteration > 1000:
            break

    # 결과 복사
    cuda.memcpy_dtoh(dist_map, dist_gpu)
    dist_map[np.isinf(dist_map)] = max_dist

    print(f"✅ CUDA BFS completed in {iteration} iterations")

    return dist_map

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