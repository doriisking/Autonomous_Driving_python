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

def build_dist_map_bf_cuda(occupancy_grid_msg, max_dist=2.0, device=None):
    """
    GPU(CUDA) 가용 시 torch로 연산을 수행하는 bruteforce 거리맵 계산.
    - occupancy_grid_msg: nav_msgs/OccupancyGrid
    - max_dist: 최대 거리 (m)
    - device: None이면 자동으로 'cuda' 사용 가능 시 cuda, 아니면 cpu 선택
    return: dist_map (numpy array, dtype=float32) [m]
    """

    # --- 1. 기본 정보 추출 ---
    width  = occupancy_grid_msg.info.width
    height = occupancy_grid_msg.info.height
    res    = occupancy_grid_msg.info.resolution
    data_np = np.array(occupancy_grid_msg.data, dtype=np.int8).reshape(height, width)

    # 자동 디바이스 선택
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 작은 맵 / CUDA가 없으면 기존 numpy 구현으로 처리 (안정성)
    if device.type == "cpu":
        # 기존 numpy bruteforce (unchanged)
        dist_sq = np.full((height, width), max_dist * max_dist, dtype=np.float32)
        l = int(math.ceil(max_dist / res))

        for j in range(width):          # x index
            for k in range(height):     # y index
                if data_np[k, j] > 0:
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

    # --- GPU 경로 (torch) ---
    # 장애물 좌표 수집
    obs_coords = np.argwhere(data_np > 0)  # shape: (N_obs, 2) with [row=y, col=x]
    if obs_coords.size == 0:
        # 장애물 없음: 전부 max_dist
        return np.full((height, width), float(max_dist), dtype=np.float32)

    # 텐서 초기화
    # dist_sq: 초기값 = max_dist^2
    dist_sq = torch.full((height, width), float(max_dist * max_dist), dtype=torch.float32, device=device)

    # 그리드 좌표 텐서 (x,y)
    # X: shape (height, width) -> x index for each cell (column index)
    # Y: shape (height, width) -> y index for each cell (row index)
    # 생성은 한 번만 (GPU에 올림)
    yy = torch.arange(0, height, dtype=torch.float32, device=device).unsqueeze(1)  # (H,1)
    xx = torch.arange(0, width,  dtype=torch.float32, device=device).unsqueeze(0)  # (1,W)
    # broadcast to (H,W) when used: (yy - y_obs) etc.

    res_sq = float(res * res)

    # GPU 에서 연산 (장애물마다 전체 그리드에 대해 거리 제곱 계산 및 최소치 갱신)
    # torch.no_grad()로 그래디언트 비활성화
    with torch.no_grad():
        for (y_obs, x_obs) in obs_coords:
            # dx = (X - x_obs), dy = (Y - y_obs)
            dx2 = (xx - float(x_obs))**2      # shape (1, W)
            dy2 = (yy - float(y_obs))**2      # shape (H, 1)
            # d2_2d = (dx2 broadcast + dy2 broadcast) * res_sq  => (H,W)
            d2 = (dy2 + dx2) * res_sq         # broadcasting
            # dist_sq = min(dist_sq, d2)
            dist_sq = torch.minimum(dist_sq, d2)

    # sqrt 및 clamp, CPU로 복사
    dist_map = torch.sqrt(dist_sq)
    dist_map = torch.clamp(dist_map, max=float(max_dist))
    dist_map_cpu = dist_map.cpu().numpy().astype(np.float32)

    return dist_map_cpu
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