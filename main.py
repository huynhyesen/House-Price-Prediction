import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("data.csv")
row, col = data.shape

x = data.drop(columns = [data.columns[col - 1]])
y = data[data.columns[col - 1]]

train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.1)

x.hist(bins=20, figsize=(10, 8))
plt.tight_layout()
plt.savefig('histogram_plot.png')
plt.close()