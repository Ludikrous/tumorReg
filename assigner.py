#  Importing libraries for machine learning algorithm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
    return line

def AIholder(algorithm):
    def wrapper(*args, **kwargs):
        func = algorithm(*args, **kwargs)
        func.fit(train_features, train_labels)
        predicted_labels = func.predict(test_features)
        logfile.write( str(accuracy_score(test_labels,predicted_labels)) + ', ')
        return
    return wrapper

#for k in range(1,100,1):
#    neigh = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
#    neigh.fit(train_features, train_labels)
#    predicted_labels = neigh.predict(test_features)
#    logger(logfile, k, accuracy_score(test_labels,predicted_labels))

@AIholder
def ai(logfile, train_features, train_labels, test_features, test_labels, k):
    return KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')

kmax = 101
logfile.write(headerline(kmax) + '\n')
for counter in range(5):
    for k in range(1,kmax):
        train_features,test_features, train_labels, test_labels = train_test_split(features,labels, test_size=0.33)
        ai(logfile, train_features, train_labels, test_features, test_labels, k)
    logfile.write('\n')
