# Nhập các thư viện cần thiết
from __future__ import division, print_function, unicode_literals
import numpy as np
from scipy.spatial.distance import cdist
np.random.seed(22)  # Đặt hạt giống ngẫu nhiên cho thư viện numpy
from scipy import sparse

def convert_labels(y, C):
    """
    Chuyển đổi mảng nhãn 1 chiều thành dạng ma trận trong đó mỗi cột biểu diễn một phần tử trong y.
    Trong cột thứ i của Y, chỉ có một phần tử khác không ở vị trí y[i], và nó bằng 1.
    """
    Y = sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y

def softmax(Z):
    """
    Tính giá trị softmax cho từng tập điểm số trong Z.
    Mỗi cột của Z là một tập điểm số.
    """
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Trừ đi giá trị lớn nhất để tránh số lớn
    A = e_Z / e_Z.sum(axis=0)
    return A

# Đầu vào: ma trận x, w ban đầu
def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    """
    Thực hiện hồi quy softmax.
    X: ma trận dữ liệu
    y: nhãn
    W_init: trọng số khởi tạo
    eta: tốc độ học
    tol: ngưỡng dừng
    max_count: số lần lặp tối đa
    """
    W = [W_init]
    C = W_init.shape[1]  # Số lớp
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        mix_id = np.random.permutation(N)  # Xáo trộn dữ liệu
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)  # Lấy mẫu thứ i
            yi = Y[:, i].reshape(C, 1)  # Lấy nhãn thứ i
            ai = softmax(np.dot(W[-1].T, xi))  # Dự đoán nhãn
            W_new = W[-1] + eta * xi.dot((yi - ai).T)  # Cập nhật trọng số
            count += 1
            # Kiểm tra sự hội tụ
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W

# Tạo dữ liệu giả lập
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
# Sinh ra 500 điểm xoay quanh mỗi 3 điểm: [2, 2], [8, 3], [3, 6]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# Ghép nối tất cả các điểm dữ liệu và thêm một thuộc tính độ chệch (bias)
X = np.concatenate((X0, X1, X2), axis=0).T
X = np.concatenate((np.ones((1, 3*N)), X), axis=0)
C = 3

# Tạo nhãn cho dữ liệu
y = original_label = np.asarray([0]*N + [1]*N + [2]*N).T
print(X.T)
print(y)

# Tốc độ học và khởi tạo trọng số
eta = .05
W_init = np.random.randn(X.shape[0], C)

# Huấn luyện mô hình
W = softmax_regression(X, original_label, W_init, eta)

# Dự đoán cho một điểm dữ liệu mới
xi = np.array([1.0, 1.90805008, 0.53664935], dtype=float)
ai = softmax(np.dot(W[-1].T, xi))
print(ai)

# Dự đoán cho một điểm dữ liệu mới khác
xnew = np.array([1, 6.0, 2.0], dtype=float)
a = softmax(np.dot(W[-1].T, xnew))
print(a)
