import pandas as pd
import numpy as np

missing_values = ["?"]
data = pd.read_csv("data/adult.data", na_values=missing_values, delimiter=", ", engine="python")
print(data.columns)
print(data.head(10))

# Filling in missing values with the mode
# TODO: decide median vs. mode
for col in data.columns:
    count = data[col].isnull().sum()
    mode = data[col].mode()

    if count > 0:
        data[col].fillna(mode, inplace=True)

    print(col, count, mode[0])

