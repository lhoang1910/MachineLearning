# Import các thư viện cần thiết
from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu chiều cao (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
# Dữ liệu cân nặng (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Thêm cột 1 vào X để tính hệ số tự do (bias)
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)

# Tính các hệ số của đường thẳng hồi quy
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)  # Sử dụng giả nghịch đảo để giải phương trình tuyến tính
print('w = ', w)

w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2, endpoint=True)
y0 = w_0 + w_1*x0

y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

# print ('Dự đoán cân nặng của người có chiều cao 155 cm: %.2f (kg), số liệu thật: 52 (kg)'  %(y1))
# print ('Dự đoán cân nặng của người có chiều cao 160 cm: %.2f (kg), số liệu thật: 56 (kg)'  %(y2))

# Vẽ dữ liệu
# plt.plot(X, y, 'ro')  # Dữ liệu thực: đỏ ('ro')
# # Thiết lập giới hạn cho trục x và y
# plt.axis([140, 190, 45, 75])
# plt.xlabel('Height (cm)')
# plt.ylabel('Weight (kg)')
# plt.show()
#w0 = -33