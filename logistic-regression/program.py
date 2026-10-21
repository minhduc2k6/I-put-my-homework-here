import numpy as np
import pandas as pd #đọc CSV, mã hóa data
import matplotlib.pyplot as plt #vẽ đồ thị

from sklearn.model_selection import train_test_split
#chia data set thành train/test set
from sklearn.preprocessing import StandardScaler
#chuẩn hóa dữ liệu
from sklearn.linear_model import LogisticRegression
#tính logistic từ thư viện
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
#các hàm đánh giá

path = 'student_performance_prediction_cleaned.csv'

#---------đọc và mã hóa dữ liệu từ CSV---------
def load_data(path):
    data = pd.read_csv(path)

    #mã hóa label passed/failed thành 1/0
    data['Passed'] = data['Passed'].map({'Yes': 1, 'No': 0})
    #mã hóa Participation in Extracurricular Activities
    data['Participation in Extracurricular Activities'] = data['Participation in Extracurricular Activities'].map({'Yes': 1, 'No': 0})

    #mã hóa Parent Education Level thành one-hot
    for level in ['Associate', 'Bachelor', 'Master', 'Doctorate']:
        data['is_' + level] = (data['Parent Education Level'] == level).astype(int)

    feature_names = [
        'Study Hours per Week', 'Attendance Rate', 'Previous Grades',
        'Participation in Extracurricular Activities',
        'is_Associate', 'is_Bachelor',
        'is_Master', 'is_Doctorate',
    ]
    x = data[feature_names].values
    y = data['Passed'].values
    return x, y, feature_names

x, y, feature_names = load_data(path)

#------chia data thành train/test set-----
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

#print(x_train.shape, x_test.shape)

#------chuẩn hóa dữ liệu-----
x_train = x_train.astype(float)
x_test = x_test.astype(float)

for j in range(3):   # lặp qua 3 cột: Study Hours per Week, Attendance Rate, Previous Grades
    mean_j = x_train[:, j].mean()
    std_j = x_train[:, j].std()

    x_train[:, j] = (x_train[:, j] - mean_j) / std_j
    x_test[:, j] = (x_test[:, j] - mean_j) / std_j

#print(x_train[:, 0:3].mean(axis=0))   # kỳ vọng P=0
#print(x_train[:, 0:3].std(axis=0))    # kỳ vọng P=1
#print(x_test[:, 0:3].mean(axis=0))    # kỳ vọng P=0, nhưng không chính xác bằng 0, bình thường

#------Bias trick-----
def bias_trick(x):
    students = x.shape[0]
    column1 = np.ones((students, 1))   # ma trận đặc trưng 1
    xNew = np.hstack((column1, x))       # ghép 2 mảng theo chiều ngang
    return xNew.T                         # ma trận đặc trưng (chuyển vị)

X_train = bias_trick(x_train)
X_test = bias_trick(x_test)

def calS(w, X):
    return np.dot(w.T, X)

# #test with random weight
# np.random.seed(42)
# wTest = np.random.randn(X_train.shape[0], 1)    # 9 so, xep thanh mot cot

# s = calS(wTest, X_train)
# print(s.shape)                   # ky vong (1, 30400)

# tong = 0
# for j in range(X_train.shape[0]):
#     tong = tong + wTest[j, 0] * X_train[j, 0]

# print(tong, s[0, 0])             # 2 cái này và cái trên pahir bằng nhau

def sigmoid(s):
    return 1 / (1 + np.exp(-s)) 

#------Loss function-----
def calLoss(w, X, y):
    z = sigmoid(calS(w, X))        # xac suat cua moi hoc sinh, dang (1, N)
    N = X.shape[1]

    tong = 0
    for i in range(N):
        zi = z[0, i]
        yi = y[i]
        tong = tong + yi * np.log(zi) + (1 - yi) * np.log(1 - zi)

    return -tong
w_zero = np.zeros((X_train.shape[0], 1))
print(calLoss(w_zero, X_train, y_train))

def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # xáo trộn thứ tự học sinh
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1
            # điều kiện dừng
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
eta = 0.01
d = X_train.shape[0]
w_init = np.random.randn(d, 1)
 
w = logistic_sigmoid_regression(X_train, y_train, w_init, eta, max_count=300000)
w_final = w[-1]
 
N_train = X_train.shape[1]
print('\nSố lần cập nhật w:', len(w) - 1)
print('Loss trung bình mỗi học sinh (train):',
      calLoss(w_final, X_train, y_train) / N_train)


# Hoãn thiện và sẽ đồ thị
# 1. Đánh giá mô hình trên tập Test
s_test = calS(w_final, X_test)       # Tính s = W^T * X cho tập test
z_test = sigmoid(s_test)             # Xác suất dự đoán (từ 0 đến 1)

# Áp dụng ngưỡng cứng 0.5: >= 0.5 là Đỗ (1), < 0.5 là Trượt (0)
y_pred = (z_test >= 0.5).astype(int).flatten()

# In ra độ chính xác
print('Accuracy trên tập test (Tự code SGD):', accuracy_score(y_test, y_pred))
print('ROC-AUC trên tập test (Tự code SGD):', roc_auc_score(y_test, z_test.flatten()))


# 2. Vẽ đồ thị Sigmoid
# Lấy mảng s 1 chiều để dễ vẽ
s_test_1d = s_test.flatten()

# Tạo một dải giá trị s liên tục (từ min đến max của s_test) để vẽ đường cong cho mượt
s_range = np.linspace(np.min(s_test_1d) - 1, np.max(s_test_1d) + 1, 500)
sigmoid_range = sigmoid(s_range)

plt.figure(figsize=(10, 6))

#sigmoid curve
plt.plot(s_range, sigmoid_range, color='red', linewidth=2.5, label='Đường cong Sigmoid: $\sigma(s)$')

# Chấm các điểm dữ liệu thực tế của tập test lên đồ thị
# Passed (y=1)
plt.scatter(s_test_1d[y_test == 1], y_test[y_test == 1], 
            color='blue', alpha=0.3, label='Thực tế: Đỗ (1)', marker='o')
# Failed (y=0)
plt.scatter(s_test_1d[y_test == 0], y_test[y_test == 0], 
            color='orange', alpha=0.3, label='Thực tế: Trượt (0)', marker='x')

# Kẻ vạch phân chia ngưỡng 0.5 và vạch s=0
plt.axhline(0.5, color='black', linestyle='--', alpha=0.6, label='Ngưỡng phân lớp (0.5)')
plt.axvline(0, color='black', linestyle='--', alpha=0.6)

# Trang trí đồ thị
plt.xlabel('Tổ hợp tuyến tính $s = W^T X$')
plt.ylabel('Xác suất dự đoán Đỗ')
plt.title('Đồ thị Logistic Regression (Sigmoid) trên tập dữ liệu Sinh viên')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()