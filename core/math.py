def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """实现可整除功能。

    该函数将通道数四舍五入到最接近能被除数整除的值。此功能源自原始TensorFlow代码库，确保所有层的通道数都能被除数整除。
    具体实现可参考：https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa.

    Args:
        value (int): 原始通道数。
        divisor (int): 用于整除通道数的除数。
        min_value (int): 输出通道的最小值。默认值：None，表示最小值等于除数。
        min_ratio  (float): 四舍五入后通道数与原始通道数的最小比率。
            默认值：0.9。

    Returns:
        整型：调整后的输出通道数。
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value
