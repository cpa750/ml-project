import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cluster import KMeans
import seaborn as sb
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier


def normalise_continuous(df):
    scaler = StandardScaler()
    res = df.copy(deep=True)
    columns = ["age", "fnlwgt", "education-num",
               "capital-gain", "capital-loss", "hours-per-week"]
    res[columns] = scaler.fit_transform(df[columns])
    return res


def normalise_all(df):
    scaler = StandardScaler()
    res = df.copy(deep=True)
    res[res.columns[0:14]] = scaler.fit_transform(df[df.columns[0:14]])
    return res


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
        data[col].fillna(mode.values[0], inplace=True)

    print(col, count, mode[0])

print("Done filling null values...\n")

# Checking that na values were filled correctly
for col in data.columns:
    print(col, data[col].isnull().sum())


def show_crosstab_plots(col):
    labels = ["age", "workclass", "education", "occupation",
              "hours-per-week", "marital-status", "relationship",
              "race", "sex", "capital-gain", "capital-loss"]
    for label in labels:
        pd.crosstab(data[label], data[col], normalize="index").plot.bar(figsize=(15, 5), stacked=True)
        plt.show()


def show_box_plots():
    labels = ["age", "hours-per-week", "capital-gain", "capital-loss"]
    for label in labels:
        data[label].plot(kind="box")
        plt.show()


def show_bar_plots():
    labels = ["workclass", "education", "occupation",
              "marital-status", "relationship",
              "race", "sex"]
    for label in labels:
        data[label].value_counts().plot(kind="bar")
        plt.show()


# show_crosstab_plots("income")
# show_box_plots()
# show_pie_plots()

# data["income"].value_counts().plot(kind="pie", autopct='%1.1f%%')
# plt.show()

def encode(df):
    encoder = LabelEncoder()
    res = df.copy(deep=True)
    for column in df.columns[0:14]:
        res[column] = encoder.fit_transform(df[column])
    return res


encoded = encode(data)

distortions = []
"""for i in range(4, 25, 2):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit_predict(data)
    distortions.append(km.inertia_)"""

"""plt.plot(range(4, 26, 2), distortions, marker='x')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()"""

km = KMeans(n_clusters=17, random_state=0)
clusters = km.fit_predict(encoded[encoded.columns[0:14]])

data["cluster-id"] = km.predict(encoded[encoded.columns[0:14]])
print(data.head(10))

data = normalise_continuous(data)
encoded = normalise_all(encoded)
print(data.head(10))
print(encoded.head(10))

array = encoded.values
X = array[:, 0:14]
Y = array[:, 14]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size=validation_size)

models = []
models.append(
    ("LDA", LinearDiscriminantAnalysis(), [{'solver': ['svd', 'lsqr', 'eigen']}])
)

best_models = []
for model in models:
    gs = GridSearchCV(model[1], model[2], scoring='roc_auc', cv=10)
    gs = gs.fit(X_train, Y_train)
    best_model = gs.best_estimator_

    best_model.fit(X_train, Y_train)
    predictions = best_model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    best_models.append((best_model, accuracy_score(Y_validation, predictions)))

ada: RandomForestClassifier = RandomForestClassifier(n_estimators=300)

ada.fit(X_train, Y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_validation)
ada_accuracy_train = accuracy_score(Y_train, y_train_pred)
ada_accuracy_test = accuracy_score(Y_validation, y_test_pred)
print("AdaBoost train accuracy: ", ada_accuracy_train)
print("AdaBoost test accuracy: ", ada_accuracy_test)
