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



def build_dist_map_bf_cuda(occupancy_grid_msg, max_dist=2.0):
    """
    Bruteforce 거리맵 (CUDA 커널 사용, pycuda SourceModule 기반)
    - occupancy_grid_msg: nav_msgs/OccupancyGrid
    - max_dist: 최대 거리 [m]
    returns: dist_map (numpy.float32) [m], shape (height, width)
    """

    # --- 1) 기본 정보 추출 ---
    width  = int(occupancy_grid_msg.info.width)
    height = int(occupancy_grid_msg.info.height)
    res    = float(occupancy_grid_msg.info.resolution)
    data_np = np.array(occupancy_grid_msg.data, dtype=np.int8).reshape((height, width))

    # 빈 장애물 맵이면 전부 max_dist
    obs_positions = np.argwhere(data_np > 0).astype(np.int32)  # shape (N_obs, 2) -> (row(y), col(x))
    n_obs = obs_positions.shape[0]
    if n_obs == 0:
        return np.full((height, width), float(max_dist), dtype=np.float32)

    # --- CPU 폴백 함수 (원본 bruteforce) ---
    def cpu_bruteforce():
        dist_sq = np.full((height, width), (max_dist * max_dist), dtype=np.float32)
        l = int(math.ceil(max_dist / res))
        for j in range(width):          # x index
            for k in range(height):     # y index
                if data_np[k, j] > 0:   # obstacle at (j,k)
                    for x in range(max(0, j - l), min(width, j + l + 1)):
                        for y in range(max(0, k - l), min(height, k + l + 1)):
                            dx = (x - j)
                            dy = (y - k)
                            d2 = (dx * dx + dy * dy) * (res * res)
                            if d2 < dist_sq[y, x]:
                                dist_sq[y, x] = d2
        dist_map = np.sqrt(dist_sq)
        dist_map[dist_map > max_dist] = max_dist
        return dist_map.astype(np.float32)

    # --- 2) pycuda 사용 가능성 체크 ---
    if not pycuda_available:
        # pycuda가 없으면 CPU 버전 반환
        print("[build_dist_map_bruteforce_cuda] pycuda not available -> CPU fallback")
        return cpu_bruteforce()

    # --- 3) CUDA C 커널 (픽셀 스레드: 모든 장애물 순회) ---
    kernel_code = r"""
    extern "C" {
    __global__ void bruteforce_min_dist(
        const int *obs_x, const int *obs_y, const int n_obs,
        const float res_sq, const float max_d2,
        float *out_dist2, const int width, const int height)
    {
        // 2D index from block/grid
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int idx = y * width + x;

        float min_d2 = max_d2;

        // loop over obstacles
        for (int i = 0; i < n_obs; ++i) {
            float dx = (float)(x - obs_x[i]);
            float dy = (float)(y - obs_y[i]);
            float d2 = (dx * dx + dy * dy) * res_sq;
            if (d2 < min_d2) {
                min_d2 = d2;
            }
        }

        out_dist2[idx] = min_d2;
    }
    } // extern "C"
    """

    # --- 4) 커널 컴파일 및 함수 얻기 ---
    mod = SourceModule(kernel_code)
    kernel = mod.get_function("bruteforce_min_dist")

    # --- 5) GPU 메모리 할당 및 데이터 업로드 ---
    # split obstacle coords into X (col), Y (row)
    obs_y = obs_positions[:, 0].astype(np.int32)  # row (y)
    obs_x = obs_positions[:, 1].astype(np.int32)  # col (x)

    # device buffers
    obs_x_gpu = cuda.mem_alloc(obs_x.nbytes)
    obs_y_gpu = cuda.mem_alloc(obs_y.nbytes)

    # output dist squared flattened (1D)
    total_cells = width * height
    out_gpu = cuda.mem_alloc(total_cells * np.dtype(np.float32).itemsize)

    # copy to device
    cuda.memcpy_htod(obs_x_gpu, obs_x)
    cuda.memcpy_htod(obs_y_gpu, obs_y)

    # --- 6) kernel 런치 파라미터 ---
    block_x = 16
    block_y = 16
    grid_x = (width  + block_x - 1) // block_x
    grid_y = (height + block_y - 1) // block_y

    # 상수들
    res_sq = np.float32(res * res)
    max_d2 = np.float32(max_dist * max_dist)
    n_obs_i = np.int32(n_obs)
    width_i = np.int32(width)
    height_i = np.int32(height)

    # 런치 (2D 그리드)
    kernel(
        obs_x_gpu, obs_y_gpu, n_obs_i,
        res_sq, max_d2,
        out_gpu, width_i, height_i,
        block=(block_x, block_y, 1), grid=(grid_x, grid_y, 1)
    )

    # 동기화 및 결과 복사
    cuda.Context.synchronize()

    # 호스트 배열으로 복사
    dist_sq_flat = np.empty(total_cells, dtype=np.float32)
    cuda.memcpy_dtoh(dist_sq_flat, out_gpu)

    # 재구성 및 후처리
    dist_sq = dist_sq_flat.reshape((height, width))
    dist_map = np.sqrt(dist_sq)
    dist_map[np.isinf(dist_map)] = max_dist
    dist_map[dist_map > max_dist] = max_dist

    return dist_map.astype(np.float32)


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