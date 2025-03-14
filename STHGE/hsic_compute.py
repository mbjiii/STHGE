'''
    计算 HSIC 值，
    若 HSIC 值接近 0，说明变量 X 和 Y 近似独立；值越大，说明它们的相关性越强。
'''

import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

def centering_matrix(n):
    """生成 n × n 维的居中矩阵 H"""
    '''return np.eye(n) - np.ones((n, n)) / n'''
    I = torch.eye(n, device='cuda' if torch.cuda.is_available() else 'cpu')
    ones = torch.ones((n, n), device=I.device) / n
    return I - ones

def compute_hsic(X, Y, kernel='rbf', sigma=1.0):
    """
    计算 HSIC 独立性准则值
    :param X: 样本数据 (n_samples, n_features)
    :param Y: 目标变量或第二个样本数据 (n_samples, n_features)
    :param kernel: 选择 'rbf' 或 'linear' 作为核函数
    :param sigma: RBF 核的带宽参数
    :return: HSIC 值
    """
    
    # 如果输入是 PyTorch Tensor，则转换为 NumPy（保持计算精度）
    if isinstance(X, torch.Tensor):
        X = X.cpu().detach().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().detach().numpy()
    
    
    n = X.shape[0]  # 样本数
    H = centering_matrix(n)  # 计算 H 居中矩阵

    # 计算 Gram 矩阵
    if kernel == 'rbf':                                     # rbf_kernel(x,y) = exp(-gamma * ||x - y||^2), gamma = 1 / (2 * sigma^2)
        '''K = rbf_kernel(X, X, gamma=1 / (2 * sigma ** 2))
        L = rbf_kernel(Y, Y, gamma=1 / (2 * sigma ** 2))'''
        K = torch.tensor(rbf_kernel(X, X, gamma=1 / (2 * sigma ** 2)), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        L = torch.tensor(rbf_kernel(Y, Y, gamma=1 / (2 * sigma ** 2)), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    elif kernel == 'linear':                                # lin_kernel(x,y) = x^T * y
        '''K = linear_kernel(X, X)
        L = linear_kernel(Y, Y)'''
        K = torch.tensor(linear_kernel(X, X), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
        L = torch.tensor(linear_kernel(Y, Y), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError("Unsupported kernel type. Use 'rbf' or 'linear'.")

    # 计算 HSIC 值
    HSIC_value = torch.trace(K @ H @ L @ H) / (n - 1) ** 2
    
    return HSIC_value.item()























'''# 生成测试数据
np.random.seed(42)
X = np.random.rand(100, 1)  # 100 个样本
Y = np.sin(X) + 0.1 * np.random.randn(100, 1)  # 非线性相关数据

# 计算 HSIC
hsic_rbf = compute_hsic(X, Y, kernel='rbf', sigma=0.5)
hsic_linear = compute_hsic(X, Y, kernel='linear')

print("HSIC (RBF 核):", hsic_rbf)
print("HSIC (线性核):", hsic_linear)'''
