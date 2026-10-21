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
    for level in ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']:
        data['is_' + level.replace(' ', '')] = (data['Parent Education Level'] == level).astype(int)

    feature_names = [
        'Study Hours per Week', 'Attendance Rate', 'Previous Grades',
        'Participation in Extracurricular Activities',
        'is_HighSchool', 'is_Associate', 'is_Bachelor',
        'is_Master', 'is_Doctorate',
    ]
    x = data[feature_names].values
    y = data['Passed'].values
    return x, y, feature_names

x, y, feature_names = load_data(path)

#------chia data thành train/test set-----
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)

print(x_train.shape, x_test.shape)

#------chuẩn hóa dữ liệu-----
x_train = x_train.astype(float)
x_test = x_test.astype(float)

for j in range(3):   # lặp qua 3 cột: Study Hours per Week, Attendance Rate, Previous Grades
    mean_j = x_train[:, j].mean()
    std_j = x_train[:, j].std()

    x_train[:, j] = (x_train[:, j] - mean_j) / std_j
    x_test[:, j] = (x_test[:, j] - mean_j) / std_j

print(x_train[:, 0:3].mean(axis=0))   # kỳ vọng P=0
print(x_train[:, 0:3].std(axis=0))    # kỳ vọng P=1
print(x_test[:, 0:3].mean(axis=0))    # kỳ vọng P=0, nhưng không chính xác bằng 0, bình thường

#------Bias trick-----
def bias_trick(x):
    so_hoc_sinh = x.shape[0]
    cot_so_1 = np.ones((so_hoc_sinh, 1))   # ma trận đặc trưng 1
    x_moi = np.hstack((cot_so_1, x))       # ghép 2 mảng theo chiều ngang
    return x_moi.T                         # ma trận đặc trưng (chuyển vị)

X_train = bias_trick(x_train)
X_test = bias_trick(x_test)

print(X_train.shape, X_test.shape)