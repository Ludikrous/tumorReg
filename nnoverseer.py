#  Importing libraries for machine learning algorithm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as SVM
from sklearn.linear_model import LogisticRegression as LGR
logfile = open('nndata.csv', 'a')

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

def AIholder(algorithm):
    def wrapper(*args, **kwargs):
        func = algorithm(*args, **kwargs)
        func.fit(train_features, train_labels)
        predicted_labels = func.predict(test_features)
        return list(predicted_labels)
    return wrapper

@AIholder
def knn_ai(logfile, train_features, train_labels, test_features, test_labels, k):
    return KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')

@AIholder
def svm_ai(logfile, train_features, train_labels, test_features, test_labels, k):
    return SVM(C = k, kernel = 'linear', cache_size=7000)

@AIholder
def lin_ai(logfile, train_features, train_labels, test_features, test_labels, k):
    return LGR(penalty='l2', C = k, solver='liblinear', cache_size=7000)

for useless in range(6):
    train_features,test_features, train_labels, test_labels = train_test_split(features,labels, test_size=0.33)
    knn_labels = knn_ai(logfile, train_features, train_labels, test_features, test_labels, k)
    svm_labels = svm_ai(logfile, train_features, train_labels, test_features, test_labels, k)
    lin_labels = lin_ai(logfile, train_features, train_labels, test_features, test_labels, k)
    for i in range(len(test_labels)):
        logfile.write(knn_labels +','+ svm_labels +','+ lin_labels +','+ test_labels +'\n')
