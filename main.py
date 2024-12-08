import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
def split_data(data):
    row, col = data.shape
    x = data.drop(columns = [data.columns[col - 1]])
    y = data[data.columns[col - 1]]
    train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.1)
    return train_data, test_data, train_label, test_label

def normalize_data(train_data, test_data):
    scaler = MinMaxScaler()
    train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
    return train_data_scaled, test_data_scaled


data = pd.read_csv("data.csv")

train_data, test_data, train_label, test_label = split_data(data)

train_data_scaled, test_data_scaled = normalize_data(train_data, test_data)

def distance_euclidean(x1, x2):
    sum = 0
    x1, x2 = np.array(x1), np.array(x2)

    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)

def knn(train_data, train_label, test_data, k):
    distance = []

    for i in range(len(train_data)):
        distance.append([distance_euclidean(train_data.iloc[i], test_data), train_label.iloc[i]])

    distance.sort(key = lambda x : x[0])
    distance = distance[:k]

    sum_label = 0
    for i in range(len(distance)):
        sum_label += distance[i][1]
    return sum_label / k

k = knn(train_data_scaled, train_label, test_data.iloc[0], 5)

print(k)