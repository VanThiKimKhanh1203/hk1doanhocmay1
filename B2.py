import os
import numpy as np
from skimage import io
from skimage.transform import resize
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image

def check_corrupted_image(img_file):
    try:
        with Image.open(img_file) as img:
            img.verify()
            img_new = io.imread(img_file)
        return False
    except Exception as e:
        print(e)
        return True

def read_img_data(path, label, size):
    X = []
    y = []
    files = os.listdir(path)
    for img_file in files:
        if not(check_corrupted_image(os.path.join(path, img_file))):
            img = io.imread(os.path.join(path, img_file), as_gray = True)
            img = resize(img, size).flatten()
            X.append(img)
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

#hàm chuyển ảnh màn hình thành vector 1024
def convert_D_2_vector(path,label,size):
    labels = []
    img_data = []
    images = os.listdir(path)
    for img_file in images:
        if not(check_corrupted_image(os.path.join(path,img_file))):
            img_grey = io.imread(os.path.join(path,img_file), as_grey = True)
            img_vector = resize(img_grey,size).flatten
            img_data.append(img_vector)
            labels.append(label)
    return img_data, labels

#Hàm huấn luyện mô hình
def kNN_grid_search_cv(X_train, y_train):
  from math import sqrt
  m = y_train.shape[0]
  k_max = int(sqrt(m)/2)
  k_values = np.arange(start = 1, stop = k_max + 1, dtype = int)
  params = { 'n_neighbors': k_values}
  kNN = KNeighborsClassifier()
  kNN_grid = GridsearchCV(kNN, params, cv=3)
  kNN_grid.fit(X_train, y_train)
  return kNN_grid

def logistic_regression_cv(X_train, y_train):
    logistic_classifier = LogisticRegreesionCV(cv=5, solver="sag", max_iter=2000)
    logistuc_classifier.fit(X_train, y_train)
    return logistic_classifier

#Hàm đánh giá mô hình
def evaluate_model(y_test, y_pred):
  print("accuracy score: ", accuracy_score(y_test, y_pred))
  print("Balandced accuracy score: ", balandced_accuracy_score(y_test, y_pred))
  print("Haming loss: ", hamming_loss(y_test, y_pred))

def confusion_matrix(y_test, y_pred, model):
    ax1 = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt=".0f", cmap="crest")
    ax1.xaxis.tick_top()
    plt.savefig("CM.png")
    plt.show()
    plt.close()

def draw_precision_recall_curve(X_test, y_test, modals: dict):
    no_modal = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_modal, no_modal], linestyle="--", label="no modal")
    for modal in modals.keys():
        probs = modal.predict_probs(X_test)[:, 1]
        pre, rec = precision_recall_curve(y_test, probs)
        plt.plot(rec, pre, label=modals[modal])
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
    X, y = read_img_data('D:/kagglecatsanddogs_5340/PetImages/Cat','Cat', (32,32))
    X_dog, y_dog = read_img_data('D:/kagglecatsanddogs_5340/PetImages/Dog','Dog', (32,32))
    X.extend(X_dog)
    y.extend(y_dog)
    X = np.array(X)
    y = np.array(y)
    y = LabelBinarizer().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X,  y, shuffle =True, random_state = 123)
    print(X.shape)
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


if __name__ == '__main__':
    main()