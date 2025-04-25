import numpy as np
import cv2
from tqdm import tqdm
import os


def get_stain_matrix(I, beta=0.15, alpha=1):
    """
    从图像中提取染色矩阵
    :param I: 输入图像，形状为 (H, W, 3)
    :param beta: 消光系数的下限
    :param alpha: 用于确定极值的百分位数
    :return: 染色矩阵，形状为 (3, 2)
    """
    # 将图像转换到 OD 空间
    OD = -np.log((I.astype(np.float64) + 1) / 255)

    # 去除消光系数小于 beta 的像素
    ODhat = OD[(OD > beta).any(axis=1)]

    # 计算 ODhat 的协方差矩阵
    _, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # 选择前两个特征向量
    eigvecs = eigvecs[:, [1, 2]]

    # 确保特征向量指向正方向
    if eigvecs[0, 0] < 0:
        eigvecs[:, 0] *= -1
    if eigvecs[0, 1] < 0:
        eigvecs[:, 1] *= -1

    # 投影到特征向量上
    That = np.dot(ODhat, eigvecs)

    # 计算极值
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    # 计算染色向量
    v1 = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))

    # 确保 H 在前，E 在后
    if v1[0] > v2[0]:
        HE = np.array([v1, v2]).T
    else:
        HE = np.array([v2, v1]).T

    return HE


def normalize_stains(I, target_stain_matrix, maxC_target, beta=0.15, alpha=1):
    """
    对图像进行染色归一化
    :param I: 输入图像，形状为 (H, W, 3)
    :param target_stain_matrix: 目标染色矩阵，形状为 (3, 2)
    :param maxC_target: 目标染色浓度的最大值，形状为 (2,)
    :param beta: 消光系数的下限
    :param alpha: 用于确定极值的百分位数
    :return: 归一化后的图像，形状为 (H, W, 3)
    """
    # 获取输入图像的染色矩阵
    stain_matrix = get_stain_matrix(I, beta, alpha)

    # 将图像转换到 OD 空间
    OD = -np.log((I.astype(np.float64) + 1) / 255)

    # 计算染色浓度
    C = np.linalg.lstsq(stain_matrix, OD.reshape((-1, 3)).T, rcond=None)[0].T

    # 归一化染色浓度
    maxC = np.percentile(C, 99, axis=0)
    C = C / maxC[None, :] * maxC_target[None, :]

    # 重建图像
    OD_norm = np.dot(C, target_stain_matrix.T)
    I_norm = np.exp(-OD_norm).reshape(I.shape)
    I_norm = (I_norm * 255).astype(np.uint8)

    return I_norm


def stain_normalize_dataset(data_dir, output_dir, target_image_path):
    """
    对整个数据集进行染色归一化
    :param data_dir: 数据集目录
    :param output_dir: 归一化后图像的输出目录
    :param target_image_path: 目标图像的路径
    """
    # 读取目标图像
    target_image = cv2.cvtColor(cv2.imread(target_image_path), cv2.COLOR_BGR2RGB)

    # 获取目标图像的染色矩阵和最大染色浓度
    target_stain_matrix = get_stain_matrix(target_image)
    target_C = np.linalg.lstsq(target_stain_matrix, (-np.log((target_image.astype(np.float64) + 1) / 255)).reshape((-1, 3)).T, rcond=None)[0].T
    maxC_target = np.percentile(target_C, 99, axis=0)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历数据集中的所有图像
    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files, desc="Processing images"):
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

                # 进行染色归一化
                normalized_image = normalize_stains(image, target_stain_matrix, maxC_target)

                # 保存归一化后的图像
                relative_path = os.path.relpath(image_path, data_dir)
                output_path = os.path.join(output_dir, relative_path)
                output_folder = os.path.dirname(output_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                cv2.imwrite(output_path, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    data_dir = "your_dataset_directory"
    output_dir = "normalized_dataset"
    target_image_path = "target_image.jpg"

    stain_normalize_dataset(data_dir, output_dir, target_image_path)
    