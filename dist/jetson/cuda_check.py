import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# GPU 커널 코드 (C 문법)
mod = SourceModule("""
__global__ void add(float *a, float *b, float *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
""")

# 랜덤 데이터 준비
a = np.random.randn(10).astype(np.float32)
b = np.random.randn(10).astype(np.float32)
c = np.zeros_like(a)

# GPU 메모리 할당 및 복사
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# GPU 커널 함수 가져오기
add_func = mod.get_function("add")

# GPU에서 연산 실행
add_func(a_gpu, b_gpu, c_gpu, block=(10,1,1), grid=(1,1))

# 결과를 CPU로 복사
cuda.memcpy_dtoh(c, c_gpu)

print("a + b (GPU 계산 결과):", c)
print("a + b (CPU 계산 결과):", a + b)
