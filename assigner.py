#  Importing libraries for machine learning algorithm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

logfile = open('logfile.csv', 'a')

# Loading the data
# Loading the entire dataset (569 entries)
# data is a dictionary, where they keys are 'target names', 'targets', 'feature names', and 'data'. The values in the key-value pairs are arrays
data = load_breast_cancer() # Extract the target names (2 available: benign, malignant)
label_names = data['target_names'] # For each entry, which target is classified(training data)?
labels = data['target'] # creating names for each attribute (30)
feature_names = data['feature_names'] #reading the 30 features for each of the 569 entries
features = data['data'] # Splits the avilable data into testing and training datasets
train_features,test_features, train_labels, test_labels = train_test_split(features,labels, test_size=0.33, random_state = 42)

def logger(file, header, value):
    file.write(str(header) +","+ str(value) + "\n")

for k in range(1,100,1):
    neigh = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute') # Fitting/Training the model to training data
    neigh.fit(train_features, train_labels) # Make a prediction of the test_features dataset
    predicted_labels = neigh.predict(test_features) # Evaluate the accuracy
    logger(logfile, k, accuracy_score(test_labels,predicted_labels))
