import torch
import time
import random

# 랜덤 A와 B 값 생성 (-10 ~ 10 범위)
def generate_random_values(size):
    A = torch.tensor([random.uniform(-10, 10) for _ in range(size)])
    B = torch.tensor([random.uniform(-10, 10) for _ in range(size)])
    return A, B

# 모든 값을 곱하는 경우 (0 포함)
def multiply_all(A, B):
    start_time = time.time()
    result = A * B
    end_time = time.time()
    return result, end_time - start_time

# 0인 값을 건너뛰고 곱하는 경우
def multiply_skip_zeros(A, B):
    start_time = time.time()
    result = []
    for a, b in zip(A, B):
        if a == 0 or b == 0:
            continue  # A 또는 B가 0이면 곱셈을 건너뜀
        result.append(a * b)
    end_time = time.time()
    return torch.tensor(result), end_time - start_time

# 실험 설정
size = 1000000  # 요소 수
A, B = generate_random_values(size)

# 모든 값을 곱할 때 시간 측정
result_all, time_all = multiply_all(A, B)
print(f"모든 값을 곱할 때 시간: {time_all:.6f} 초")

# 0인 값을 건너뛰고 곱할 때 시간 측정
result_skip, time_skip = multiply_skip_zeros(A, B)
print(f"0인 값을 건너뛰고 곱할 때 시간: {time_skip:.6f} 초")
