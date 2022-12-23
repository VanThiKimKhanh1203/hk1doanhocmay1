import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage import io
from skimage.transform import resize
import os

# Tien xu ly du lieu
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# Huan luyen mo hinh
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
# Danh gia mo hinh
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score


PATH = "D:/kagglecatsanddogs_5340/PetImages/"


def check_corrupted_image(file):
    try:
        with Image.open(file) as img:
            if not img.verify():
                img_new = io.imread(file)

        return False
    except Exception as e:
        print(e)
        return True


def read_img_data(path, size):
    X = []
    y = []

    label = path.split("\\")[-1]
    files = os.listdir(path)
    for img_file in files:
        if not check_corrupted_image(os.path.join(path, img_file)):
            img = io.imread(os.path.join(path, img_file), as_gray=True)
            img = resize(img, size)
            img_flatten = list(img.flatten())
            X.append(img_flatten)
            y.append(label)
    return X, y


def read_img_datasets(folder_path, size):
    X = []
    y = []

    for img_folder in os.listdir(folder_path):
        X_temp, y_temp = read_img_data(os.path.join(folder_path, img_folder), size)
        X.extend(X_temp)
        y.extend(y_temp)

    return np.array(X), np.array(y)


def encode_label(y):
    lb = LabelBinarizer()
    return lb.fit_transform(y).reshape(y.shape[0], )


def count_unique_labels(y):
    unique, counts = np.unique(y, return_counts=True)
    result = dict(zip(unique, counts))
    return result


# Ham huan luyen mo hinh:

def kNN_grid_search_cv(X_train, y_train):
    from math import sqrt
    m = y_train.shape[0]
    k_max = int(sqrt(m) / 2)
    k_values = np.arange(start=1, stop=k_max + 1, dtype=int)
    params = {'n_neighbors': k_values}
    kNN = KNeighborsClassifier()
    kNN_grid = GridSearchCV(kNN, params, cv=3)
    kNN_grid.fit(X_train, y_train)
    return kNN_grid


def logistic_regression_cv(X_train, y_train):
    logistic_classifier = LogisticRegressionCV(cv=5, solver="sag", max_iter=2000)
    logistic_classifier.fit(X_train, y_train)
    return logistic_classifier


# Ham danh gia mo hinh
def evaluate_model(y_test, y_pred):
    print("accuracy score: ", accuracy_score(y_test, y_pred))
    print("Balanced accuracy score: ", balanced_accuracy_score(y_test, y_pred))
    print("Hamming loss: ", hamming_loss(y_test, y_pred))


def confusion_matrix(y_test, y_pred, model):
    ax1 = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", cmap="crest")
    ax1.xaxis.tick_top()
    plt.savefig("CM.png")
    plt.show()
    plt.close()


def test_score(y_test, y_pred, class_type="binary"):
    if class_type == "binary":
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        result = [accuracy, precision, recall, f1]
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        result = [accuracy, precision, recall, f1]

    return result


def test_table(modal: dict):
    result = {
        "Scores": ["accuracy", "precision", "recall", "f1"]
    }
    for modal in modals:
        modal_score = test_score(modals[modal][0], modals[modal][1])

    return pd.DataFrame(result)


def draw_precision_recall_curve(X_test, y_test, modals: dict):
    no_modal = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_modal, no_modal], linestyle="--", label="no modal")
    for modal in modals.keys():
        probs = modal.predict_probs(X_test)[:, 1]
        pre, rec = precision_recall_curve(y_test, probs)
        plt.plot(rec, pre, label=modals[modal])
    print("OK...")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("PRC.png")
    plt.show()
    plt.close()


def drawROC(X_test, y_test, modals: dict):
    for modal in modals.keys():
        probs = modal.predict_probs(X_test)[:, -1]
        auc = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("ROC.png")
    plt.show()
    plt.close()

def main():

    # Đọc dữ liệu ảnh,nhãn từ các folder:
    X, y = read_img_datasets(PATH, size=(32, 32))

    # Mã hóa nhãn lớp:
    y = encode_label(y)

    # Phân chia tập train, test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

    # Huấn luyện mô hình:

    kNN_classifier = kNN_grid_search_cv(X_train, y_train)
    logistic_classifier = logistic_regression_cv(X_train, y_train)

    # Dự đoán kết quả:
    y_pred_kNN = kNN_classifier.predict(X_test)
    y_pred_logistic = logistic_classifier.predict(X_test)

    # Đánh giá mô hình:
    evaluate_model(y_test, y_pred_kNN)
    evaluate_model(y_test, y_pred_logistic)

    print(test_table({
        "kNN": [y_test, y_pred_kNN],
        "Logistic Regression": [y_test, y_pred_logistic]
    }))

    draw_precision_recall_curve(X_test, y_test, {
        kNN_classifier: "kNN",
        logistic_classifier: "Logistic Regression"
    })
    draw_ROC(X_test, y_test, {
        kNN_classifier: "kNN",
        logistic_classifier: "Logistic Regression"
    })


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

