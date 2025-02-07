import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

# Hàm chia và chuẩn hóa dữ liệu

def split_train_test(data, labels, test_ratio=0.1):
    combined_data = np.column_stack((data, labels))
    np.random.shuffle(combined_data)
    split_index = int(len(combined_data) * (1 - test_ratio))
    train_set = combined_data[:split_index]
    test_set = combined_data[split_index:]
    return train_set[:, :-1], train_set[:, -1], test_set[:, :-1], test_set[:, -1]

def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Hàm vẽ biểu đồ trước khi chuẩn hóa

def data_before_normalization(train_data, column_names):
    train_data_df = pd.DataFrame(train_data)

    train_data_df = train_data_df.iloc[:, 1:]
    train_data_df.columns = column_names[1:]

    train_data_df.hist(bins=20, figsize=(10, 8))
    plt.suptitle("Histogram before normalization")
    plt.show()

# Hàm vẽ biểu đồ sau khi chuẩn hóa

def data_after_normalization(train_data, test_data, column_names):
    train_data_df = pd.DataFrame(train_data)
    test_data_df = pd.DataFrame(test_data)

    train_data_df = train_data_df.iloc[:, 1:]
    test_data_df = test_data_df.iloc[:, 1:]
    train_data_df.columns = column_names[1:]
    test_data_df.columns = column_names[1:]

    train_data_df.hist(bins=20, figsize=(10, 8))
    plt.suptitle("Histogram after normalization (Train Data)")
    plt.show()

    test_data_df.hist(bins=20, figsize=(10, 8))
    plt.suptitle("Histogram after normalization (Test Data)")
    plt.show()

# Thuật toán K - Nearest Neighbors

def knn(train_data, train_labels, test_data, k):
    predictions = []
    for test_instance in test_data:
        distances = []
        for i, train_instance in enumerate(train_data):
            distance = np.linalg.norm(test_instance - train_instance)
            distances.append((train_instance, train_labels[i], distance))
        distances.sort(key=lambda x: x[2])
        neighbors = [x[1] for x in distances[:k]]
        prediction = np.mean(neighbors)
        predictions.append(prediction)
    return predictions

# Hàm tính RMSE

def rmse(actual, predicted):
    return np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))

# Hàm vẽ biểu đồ RMSE theo giá trị k

def rmse_vs_k(train_data, train_labels, test_data, test_labels, max_k):
    rmse_values = []
    for k in range(1, max_k + 1):
        predictions = knn(train_data, train_labels, test_data, k)
        rmse_value = rmse(test_labels, predictions)
        print(f'RMSE for k={k}: {rmse_value}')
        rmse_values.append(rmse_value)
    plt.plot(range(1, max_k + 1), rmse_values)
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.show()

# Đường dẫn tệp
file_path = os.path.join(os.getcwd(), 'real_estate.csv')

# Đọc dữ liệu từ file CSV
data = pd.read_csv(file_path)
labels = data.iloc[:, -1]
features = data.iloc[:, :-1]

column_names = features.columns

# Chia dữ liệu
train_data, train_labels, test_data, test_labels = split_train_test(features, labels)

# Vẽ dữ liệu trước khi chuẩn hóa
data_before_normalization(train_data, column_names)

train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

# Vẽ dữ liệu sau khi chuẩn hóa
data_after_normalization(train_data, test_data, column_names)

# Vẽ biểu đồ RMSE theo giá trị k
rmse_vs_k(train_data, train_labels, test_data, test_labels, 20)
