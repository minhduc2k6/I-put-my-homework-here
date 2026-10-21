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

def load_data(path):
    data = pd.read_csv(path)

    #mã hóa label passed/failed thành 1/0
    data['passed'] = data['passed'].map({'yes': 1, 'No': 0})
    #mã hóa Participation in Extracurricular Activities
    data['Participation in Extracurricular Activities'] = data['Participation in Extracurricular Activities'].map({'yes': 1, 'No': 0})

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
    y = data['passed'].values.shape
    return x, y, feature_names

x, y, feature_names = load_data(path)
print(x.shape, y.shape)
