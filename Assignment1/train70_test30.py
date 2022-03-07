import re
import numpy
from sklearn.datasets import load_files
from nltk.corpus import stopwords

# Loading data from enron1, enron2, enron3, enron4, enron5
email_data = load_files(r"data/enron1")
X, y = email_data.data, email_data.target

email_data = load_files(r"data/enron2")
X2, y2 = email_data.data, email_data.target

email_data = load_files(r"data/enron3")
X3, y3 = email_data.data, email_data.target

email_data = load_files(r"data/enron4")
X4, y4 = email_data.data, email_data.target

email_data = load_files(r"data/enron5")
X5, y5 = email_data.data, email_data.target

# Conflating all the data into one dataset
X += X2 + X3 + X4 + X5
y = numpy.concatenate([y, y2, y3, y4, y5])

documents = []

for sen in range(0, len(X)):
    # Removing special and single characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Removing multiple spaces and converting to lowercase
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    document = document.lower()

    # Lemmatization
    document = document.split()
    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

# Vectorisation
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()

# Splitting dataset into 70% training and 30% testing while maintaining ham:spam ratio
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=2)
StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Naive Bayes Results
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Naive Bayes: ", accuracy_score(y_test, y_pred), "\n")

# Neural Network Results
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)
predict_test = mlp.predict(X_test)
print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))
print("Neural Network: ", accuracy_score(y_test, predict_test), "\n")

# SVM Results
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)
predict_test = clf.predict(X_test)
print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))
print("SVM: ", accuracy_score(y_test, predict_test))
