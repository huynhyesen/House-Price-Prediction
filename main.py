import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")
row, col = data.shape

x = data.drop(columns = [data.columns[col - 1]])
y = data[data.columns[col - 1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)



