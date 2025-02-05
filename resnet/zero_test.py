import time
import random

# 실수 범위 설정
min_val = -10
max_val = 10

# 곱셈을 스킵한 경우의 시간 측정 (0을 포함할 확률 추가)
'''def measure_multiplication_time_with_skip():
    start_time = time.time()
    
    for _ in range(1000000):  # 반복 횟수
        # 90% 확률로 -10 ~ 10 사이의 실수 생성, 10% 확률로 0 생성
        A = random.uniform(min_val, max_val) if random.random() > 0.1 else 0
        B = random.uniform(min_val, max_val) if random.random() > 0.1 else 0
        
        # A나 B가 0이면 곱셈을 스킵
        if A == 0 or B == 0:
            continue  # 곱셈을 스킵
        else:
            result = A * B  # 곱셈 수행
    
    end_time = time.time()
    
    return end_time - start_time'''

# 곱셈을 스킵하지 않고 항상 진행하는 함수
def measure_multiplication_time_no_skip():
    start_time = time.time()
    
    for _ in range(1000000):  # 반복 횟수
        # 90% 확률로 -10 ~ 10 사이의 실수 생성, 10% 확률로 0 생성
        A = random.uniform(min_val, max_val) if random.random() > 0.1 else 0
        B = random.uniform(min_val, max_val) if random.random() > 0.0 else 0
        
        result = A * B  # 항상 곱셈 수행
    
    end_time = time.time()
    
    return end_time - start_time


# 곱셈을 스킵하지 않은 경우의 시간 측정
time_no_skip = measure_multiplication_time_no_skip()
print(f"곱셈을 스킵하지 않은 경우 시간: {time_no_skip:.6f}초")

# 곱셈을 스킵한 경우의 시간 측정
#time_with_skip = measure_multiplication_time_with_skip()
#print(f"곱셈을 스킵한 경우 시간: {time_with_skip:.6f}초")

