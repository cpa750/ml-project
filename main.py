import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import seaborn as sns
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def cluster(dat, enc):
    kMeans = KMeans(n_clusters=17, random_state=0)
    kMeans.fit_predict(enc[enc.columns[0:14]])

    dat["cluster-id"] = kMeans.predict(enc[enc.columns[0:14]])


def encode(df, columns):
    encoder = LabelEncoder()
    res = df.copy(deep=True)
    for column in df.columns[0:columns]:
        res[column] = encoder.fit_transform(df[column])
    return res


def normalise_continuous(df):
    scaler = MinMaxScaler()
    res = df.copy(deep=True)
    columns = ["age", "fnlwgt", "education-num",
               "capital-gain", "capital-loss", "hours-per-week"]
    res[columns] = scaler.fit_transform(df[columns])
    return res


def normalise_all(df):
    scaler = MinMaxScaler()
    res = df.copy(deep=True)
    res[res.columns[0:14]] = scaler.fit_transform(df[df.columns[0:14]])
    return res


def show_bar_plots():
    labels = ["workclass", "education", "occupation",
              "marital-status", "relationship",
              "race", "sex"]
    for label in labels:
        data[label].value_counts().plot(kind="bar")
        plt.xlabel(label.capitalize())
        plt.savefig("images/" + label + ".png", bbox_inches="tight")
        plt.clf()
        plt.cla()
        plt.close()


def show_box_plots():
    labels = ["age", "hours-per-week", "capital-gain", "capital-loss"]
    for label in labels:
        data[label].plot(kind="box")
        plt.xlabel(label.capitalize())
        plt.xticks([])
        plt.savefig("images/" + label + ".png", bbox_inches="tight")
        plt.clf()
        plt.cla()
        plt.close()


def show_confusion_matrix(name, Y_val, pred):
    cm = confusion_matrix(Y_val, pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, ax=ax, cmap="Blues", fmt='g')
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    ax.set_xticklabels(["<=50K", ">50K"])
    ax.set_yticklabels(["<=50K", ">50K"])
    plt.savefig("images/" + name + "scatter plot.png")


def show_crosstab_plots(column):
    labels = ["age", "workclass", "education", "occupation",
              "hours-per-week", "marital-status", "relationship",
              "race", "sex", "capital-gain", "capital-loss"]
    for label in labels:
        pd.crosstab(data[label], data[column], normalize="index").plot.bar(figsize=(20, 5), stacked=True)
        plt.xlabel(label.capitalize())
        plt.ylabel("Ratio of <=50K income to >50K income")
        plt.savefig("images/" + label + " vs " + column + ".png", bbox_inches="tight")
        plt.clf()
        plt.cla()
        plt.close()


missing_values = ["?"]
data = pd.read_csv("data/adult.data", na_values=missing_values, delimiter=", ", engine="python")
# Filling in missing values with the mode
for col in data.columns:
    count = data[col].isnull().sum()
    mode = data[col].mode()

    if count > 0:
        data[col].fillna(mode.values[0], inplace=True)

show_crosstab_plots("income")
show_box_plots()
show_bar_plots()

data["income"].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.xlabel(None)
plt.ylabel(None)
plt.savefig("images/income.png", bbox_inches="tight")
plt.clf()
plt.cla()
plt.close()

encoded = encode(data, 15)

distortions = []
for i in range(4, 25, 2):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit_predict(encoded)
    distortions.append(km.inertia_)

plt.plot(range(4, 26, 2), distortions, marker='x')
plt.xlabel("Clusters")
plt.ylabel("Distortion")
plt.tight_layout()
plt.savefig("images/distortion.png", bbox_inches="tight")
plt.clf()
plt.cla()
plt.close()

encoded = encode(data, 14)
encoded = normalise_all(encoded)
array = encoded.values
X = array[:, 0:14]
Y = array[:, 14]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, Y, test_size=validation_size, random_state=42)

models = [
    ("LDA", LinearDiscriminantAnalysis(), [{"solver": ["svd", "lsqr", "eigen"]}]),
    ("CART", DecisionTreeClassifier(), [{"criterion": ["gini", "entropy"],
                                         "splitter": ["random", "best"],
                                         "max_depth": [1, 3, 5, 10, 15, 20, 30]}]),
    ("KNN", KNeighborsClassifier(), [{"n_neighbors": [3, 5, 10],
                                      "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                                      "leaf_size": [10, 20, 30, 40, 50]}]),
    ("GaussianNB", GaussianNB(), [{}])
]

best_models = []
for model in models:
    results = []
    accuracy = 0
    X_train_new = []
    X_val_new = []
    for i in range(5, 101, 5):
        fs = SelectPercentile(chi2, percentile=i)
        X_train_fs = fs.fit_transform(X_train, Y_train)
        scores = cross_val_score(model[1], X_train_fs, Y_train, cv=10)
        results = np.append(results, scores.mean())
        if scores.mean() > accuracy:
            X_train_new = X_train_fs
            X_val_new = fs.transform(X_validation)
            accuracy = scores.mean()

    plt.figure()
    plt.xlabel("Features Selected")
    plt.ylabel("Cross-Valuation Score")
    plt.plot(range(5, 101, 5), results)
    plt.savefig("images/" + model[0] + " Feature Selection.png")
    plt.clf()
    plt.cla()
    plt.close()

    gs = GridSearchCV(model[1], model[2], scoring='roc_auc', cv=10)
    gs = gs.fit(X_train_new, Y_train)
    best_model = gs.best_estimator_

    best_model.fit(X_train_new, Y_train)
    predictions = best_model.predict(X_val_new)
    print(model[0], ": ", model[1])
    print("Accuracy: ", accuracy_score(Y_validation, predictions))
    print("Best number of features: ", X_train_new.shape[1])

    show_confusion_matrix(model[0], Y_validation, predictions)

    best_models.append((best_model, accuracy_score(Y_validation, predictions)))

ada = AdaBoostClassifier(DecisionTreeClassifier())
gs = GridSearchCV(ada, [{"n_estimators": range(50, 1000, 50)}],
                  scoring="roc_auc", cv=10)
gs = gs.fit(X_train, Y_train)
print("AdaBoost: ", ada)
ada = gs.best_estimator_

ada.fit(X_train, Y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_validation)
ada_accuracy_train = accuracy_score(Y_train, y_train_pred)
ada_accuracy_test = accuracy_score(Y_validation, y_test_pred)
print("AdaBoost train accuracy: ", ada_accuracy_train)
print("AdaBoost test accuracy: ", ada_accuracy_test)
show_confusion_matrix("AdaBoost", Y_validation, y_test_pred)
