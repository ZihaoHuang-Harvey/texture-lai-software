import numpy as np
from skimage.feature import graycomatrix, graycoprops
"""灰度共生矩阵(GLCM)特征计算模块"""
def calculate_glcm_features(image, step_size, levels, distances, angles, properties):
    """计算图像的GLCM特征
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("输入图像必须是numpy数组")
    if levels < 2 or levels > 256:
        raise ValueError("灰度级必须在2-256之间")
    # 图像采样
    if step_size > 1:
        sampled_image = image[::step_size, ::step_size]
    else:
        sampled_image = image
        # 灰度级调整
    max_gray_value = int(np.max(sampled_image))
    if max_gray_value >= levels:
        sampled_image = np.floor(sampled_image / (max_gray_value + 1) * levels).astype(np.uint8)
    # 计算GLCM矩阵
    glcm = graycomatrix(
        sampled_image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )
    features = {}
    # 计算自相关性(自定义实现)
    if 'autocorrelation' in properties:
        autocorrelation = 0
        for d in range(len(distances)):
            for a in range(len(angles)):
                autocorrelation += _calculate_autocorrelation(glcm[:, :, d, a])
        features['autocorrelation'] = autocorrelation / (len(distances) * len(angles))
    # 计算其他GLCM特征
    for prop in properties:
        if prop == 'autocorrelation':
            continue
        feature_values = graycoprops(glcm, prop)
        features[prop] = feature_values.mean()
    return features, glcm
def _calculate_autocorrelation(glcm):
    """计算GLCM矩阵的自相关性"""
    mean_i = np.sum(np.arange(glcm.shape[0]) * glcm.sum(axis=1))
    mean_j = np.sum(np.arange(glcm.shape[1]) * glcm.sum(axis=0))
    covariance = np.sum((np.arange(glcm.shape[0]) - mean_i)[:, None] * 
                       (np.arange(glcm.shape[1]) - mean_j) * glcm)
    var_i = np.sum((np.arange(glcm.shape[0]) - mean_i) ** 2 * glcm.sum(axis=1))
    var_j = np.sum((np.arange(glcm.shape[1]) - mean_j) ** 2 * glcm.sum(axis=0))
    denominator = np.sqrt(var_i * var_j) + 1e-10
    return covariance / denominator
def save_glcm_matrix(glcm, path):
    """保存GLCM矩阵到文件"""
    np.save(path, glcm)
def calculate_features(img_path, step=1, distance=5, angle=0, grayscale=256):
    """从图像文件计算GLCM特征"""
    try:
        from PIL import Image
        img = Image.open(img_path)
        img_array = np.array(img.convert('L'))  # 转换为灰度图像
        img_rescaled = rescale_image(img_array, grayscale)  # 灰度级调整
        # 计算GLCM矩阵
        glcm_matrix = graycomatrix(
            img_rescaled,
            distances=[distance],
            angles=[np.deg2rad(angle)],  # 转换为弧度
            symmetric=True,
            normed=True
        )
        # 计算自相关性
        autocorr = _calculate_autocorrelation(glcm_matrix[:, :, 0, 0])
        # 计算其他GLCM特征
        features = {
            'contrast': graycoprops(glcm_matrix, 'contrast')[0, 0],
            'energy': graycoprops(glcm_matrix, 'energy')[0, 0],
            'homogeneity': graycoprops(glcm_matrix, 'homogeneity')[0, 0],
            'correlation': graycoprops(glcm_matrix, 'correlation')[0, 0], 
            'dissimilarity': graycoprops(glcm_matrix, 'dissimilarity')[0, 0],
            'autocorrelation': autocorr
        }
        return features, glcm_matrix
    except Exception as e:
        raise RuntimeError(f"图像处理失败: {str(e)}")
def glcm_to_image(glcm, distance_idx=0, angle_idx=0):
    """将GLCM矩阵转换为可视化图像"""
    matrix = glcm[:, :, distance_idx, angle_idx]
    # 归一化到0-255
    if matrix.max() > 0:
        normalized = (matrix / matrix.max() * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(matrix, dtype=np.uint8)
    return normalized
def rescale_image(img_array, gray_levels):
    """将图像灰度值缩放到指定级别"""
    min_val = img_array.min()
    max_val = img_array.max()
    return ((img_array - min_val) / (max_val - min_val) * (gray_levels - 1)).astype(np.uint8)
