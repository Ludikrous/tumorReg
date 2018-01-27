#  Importing libraries for machine learning algorithm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as SVM
from sklearn.linear_model import LogisticRegression as LGR

logfile = open('logfile.csv', 'a')

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

def headerline(kmax):
    line = ''
    for i in range(1, kmax):
        line = line + str(i) + ", "
    return 'pass, type, ' + line + '\n'

def AIholder(algorithm):
    def wrapper(*args, **kwargs):
        func = algorithm(*args, **kwargs)
        func.fit(train_features, train_labels)
        predicted_labels = func.predict(test_features)
        print(str(k) +' = '+ str(accuracy_score(test_labels,predicted_labels)))
        logfile.write( str(accuracy_score(test_labels,predicted_labels)) + ', ')
        return
    return wrapper

@AIholder
def knn_ai(logfile, train_features, train_labels, test_features, test_labels, k):
    return KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')

@AIholder
def svm_ai(logfile, train_features, train_labels, test_features, test_labels, k):
    return SVM(C = k, kernel = 'linear', cache_size=7000)

@AIholder
def lin_ai(logfile, train_features, train_labels, test_features, test_labels, k):
    return LGR(penalty='l2', C = k, solver='liblinear')


kmax = 101

print("---------------> testing knn...")
logfile.write(headerline(kmax))
for counter in range(10):
    train_features,test_features, train_labels, test_labels = train_test_split(features,labels, test_size=0.33)
    print("---------------> testing knn...")
    logfile.write(str(counter) + ',knn,')
    for k in range(1,kmax):
        knn_ai(logfile, train_features, train_labels, test_features, test_labels, k)
    logfile.write('\n')
    print("---------------> testing svm...")
    logfile.write(str(counter) + ',svm,')
    for k in range(1,kmax):
        svm_ai(logfile, train_features, train_labels, test_features, test_labels, k)
    logfile.write('\n')
    print("---------------> testing linear...")
    logfile.write(str(counter) + ',lin,')
    for k in range(1,kmax):
        lin_ai(logfile, train_features, train_labels, test_features, test_labels, k)
    logfile.write('\n')
    logfile.write('\n')
