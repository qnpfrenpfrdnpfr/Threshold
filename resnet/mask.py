def apply_mask(x, threshold=0.0295):
    """
    주어진 입력 텐서 x에 마스크를 적용하여 임계값 이상인 채널만 남깁니다.

    Args:
        x (torch.Tensor): 입력 텐서, [batch_size, num_channels, height, width]
        threshold (float): 채널을 선택할 임계값

    Returns:
        torch.Tensor: 마스크가 적용된 텐서
    """
    # 각 채널의 평균값 계산
    avg_values = x.mean(dim=[2, 3], keepdim=True)  # [batch_size, num_channels, 1, 1]

    # 임계값 이상인 채널을 선택하여 마스크 생성
    make_zero = avg_values >= threshold  # [batch_size, num_channels, 1, 1]

    # 마스크를 입력 텐서에 바로 적용하여 반환
    return x * make_zero  # 마스크 적용된 텐서 반환