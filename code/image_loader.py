# 图像加载和预处理工具模块
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import numpy as np
def load_image(image_path):
    # 加载并转换图像为灰度图
    image = imread(image_path)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = rgb2gray(image[..., :3])
        else:
            image = rgb2gray(image)
    return img_as_ubyte(image)
# 将图像灰度值重新量化到指定级别
def rescale_image(image, levels):
    if isinstance(levels, int):
        if levels < 2 or levels > 256:
            raise ValueError("灰度级必须为2-256之间的整数")
        bins = np.linspace(0, 255, levels, dtype=np.uint8)
    elif isinstance(levels, (list, np.ndarray)):
        if len(levels) < 2:
            raise ValueError("至少需要2个灰度级")
        if not all(0 <= x <= 255 for x in levels):
            raise ValueError("灰度值必须在0-255范围内")
        if not np.all(np.diff(levels) > 0):
            raise ValueError("灰度值必须递增排列")
        bins = np.array(levels, dtype=np.uint8)
    else:
        raise TypeError("输入类型无效，需为整数或列表")
    scaled = rescale_intensity(image.astype(float), out_range='image')
    scaled = np.clip(scaled, 0, 255)
    quantized = np.digitize(scaled, bins[:-1], right=True)
    result = bins[quantized].astype(np.uint8)
    if np.max(result) > 255 or np.min(result) < 0:
        raise ValueError("量化结果超出0-255范围")
    return result
# 重采样图像到目标尺寸
def resample_image(image, target_size):
    resampled = resize(
        image, target_size,
        order=3,
        anti_aliasing=True,
        preserve_range=True
    )
    return np.round(resampled).astype('uint8')
