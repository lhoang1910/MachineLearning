import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(22)  # Đặt seed cho hàm sinh số ngẫu nhiên

def sigmoid(s):
    return 1/(1 + np.exp(-s))  # Định nghĩa hàm sigmoid

def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]  # Khởi tạo trọng số ban đầu
    count = 0  # Đếm số lần cập nhật trọng số
    check_w_after = 20  # Kiểm tra sự thay đổi của trọng số sau mỗi 20 lần lặp

    while count < max_count:  # Lặp cho đến khi đạt số lần tối đa
        mix_id = np.random.permutation(N)  # Xáo trộn dữ liệu
        for i in mix_id:  # Duyệt qua từng điểm dữ liệu
            xi = X[:, i].reshape(d, 1)  # Lấy mẫu dữ liệu thứ i
            yi = y[i]  # Lấy nhãn của mẫu dữ liệu thứ i
            zi = sigmoid(np.dot(w[-1].T, xi))  # Tính giá trị dự đoán với hàm sigmoid
            w_new = w[-1] + eta*(yi - zi)*xi  # Cập nhật trọng số
            count += 1  # Tăng biến đếm

            if count % check_w_after == 0:  # Kiểm tra tiêu chuẩn dừng
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w  # Trả về trọng số nếu đủ điều kiện dừng
            w.append(w_new)
    return w

# Tạo dữ liệu mô phỏng
means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

# Vẽ dữ liệu
# plt.plot(X0[:, 0], X0[:, 1], 'bs', markersize=8, alpha=1)
# plt.plot(X1[:, 0], X1[:, 1], 'ro', markersize=8, alpha=1)
# plt.axis('equal')
# plt.ylim(0, 4)
# plt.xlim(0, 5)

# # Ẩn các đánh dấu trên trục
# cur_axes = plt.gca()
# cur_axes.axes.get_xaxis().set_ticks([])
# cur_axes.axes.get_yaxis().set_ticks([])

# plt.xlabel('$x_1$', fontsize=20)
# plt.ylabel('$x_2$', fontsize=20)
# plt.savefig('logistic_2d.png', bbox_inches='tight', dpi=300)
# plt.show()

# Chuẩn bị dữ liệu cho hồi quy logistic
# Nối x0 x1 = x ngang 
X = np.concatenate((X0, X1), axis=0).T  # Gộp và chuyển vị dữ liệu
y = np.concatenate((np.zeros((1, N)), np.ones((1, N))), axis=1).T  # Tạo nhãn
X = np.concatenate((np.ones((1, 2*N)), X), axis=0)  # Thêm cột 1 vào X

# Thiết lập tham số và khởi tạo trọng số
eta = .05  # Tốc độ học
d = X.shape[0]  # Số chiều dữ liệu
w_init = np.random.randn(d, 1)  # Khởi tạo trọng số ngẫu nhiên

# Thực hiện hồi quy logistic
w = logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000)
print(w[-1])  # In trọng số sau cùng
