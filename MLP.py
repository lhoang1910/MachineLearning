# Hỗ trợ cả Python 2 và Python 3
from __future__ import division, print_function, unicode_literals
import math
import numpy as np

# Định nghĩa các thông số cơ bản
N = 100 # Số điểm dữ liệu cho mỗi lớp
d0 = 2 # Số chiều của dữ liệu
C = 3 # Số lượng lớp
X = np.zeros((d0, N*C)) # Ma trận dữ liệu (mỗi hàng là một ví dụ)
y = np.zeros(N*C, dtype='uint8') # Nhãn của các lớp

# Tạo dữ liệu giả lập
for j in range(C):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # Bán kính
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # Góc theta
  X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
  y[ix] = j

# In dữ liệu và nhãn ra màn hình
# print(X.T)
# print(y)

# Định nghĩa hàm softmax
def softmax(V):
  e_V = np.exp(V - np.max(V, axis=0, keepdims=True))
  Z = e_V / e_V.sum(axis=0)
  return Z

# Chuyển đổi nhãn sang dạng one-hot
from scipy import sparse
def convert_labels(y, C=3):
  Y = sparse.coo_matrix((np.ones_like(y),
                         (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
  return Y

# Hàm tính chi phí (loss function)
def cost(Y, Yhat):
  return -np.sum(Y * np.log(Yhat)) / Y.shape[1]

# Khởi tạo kích thước cho các lớp
d0 = 2
d1 = h = 20 # Kích thước của lớp ẩn
d2 = C = 3

# Khởi tạo tham số ngẫu nhiên
W1 = 0.01*np.random.randn(d0, d1)
b1 = np.zeros((d1, 1))
W2 = 0.01*np.random.randn(d1, d2)
b2 = np.zeros((d2, 1))

# Chuyển đổi nhãn và đặt các thông số học
Y = convert_labels(y, C)
N = X.shape[1]
eta = 1 # Tốc độ học

# Quá trình huấn luyện
for i in range(10000):
    # Quá trình lan truyền tiến (feedforward)
    Z1 = np.dot(W1.T, X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = np.dot(W2.T, A1) + b2
    Yhat = softmax(Z2)

    # In ra mất mát sau mỗi 1000 lần lặp
    if i % 1000 == 0:
        loss = cost(Y, Yhat)
        print("iter %d, loss: %f" %(i, loss))

    # Lan truyền ngược (backpropagation)
    E2 = (Yhat - Y )/N
    dW2 = np.dot(A1, E2.T)
    db2 = np.sum(E2, axis=1, keepdims=True)
    E1 = np.dot(W2, E2)
    E1[Z1 <= 0] = 0 # Đạo hàm của ReLU
    dW1 = np.dot(X, E1.T)
    db1 = np.sum(E1, axis=1, keepdims=True)

    # Cập nhật gradient
    W1 += -eta*dW1
    b1 += -eta*db1
    W2 += -eta*dW2
    b2 += -eta*db2

# In ra các tham số sau khi huấn luyện
print(W1)
print(b1)
print(W2)
print(b2)

# Đánh giá độ chính xác trên tập huấn luyện
Z1 = np.dot(W1.T, X) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
predicted_class = np.argmax(Z2, axis=0)
print('training accuracy: %.2f %%' % (100*np.mean(predicted_class == y)))
