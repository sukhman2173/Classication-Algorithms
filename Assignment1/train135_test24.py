import re
import numpy
from sklearn.datasets import load_files
from nltk.corpus import stopwords

# Loading data from enron1, enron2, enron3, enron4, enron5
email_data = load_files(r"data/enron1")
X1, y1 = email_data.data, email_data.target

email_data = load_files(r"data/enron2")
X2, y2 = email_data.data, email_data.target

email_data = load_files(r"data/enron3")
X3, y3 = email_data.data, email_data.target

email_data = load_files(r"data/enron4")
X4, y4 = email_data.data, email_data.target

email_data = load_files(r"data/enron5")
X5, y5 = email_data.data, email_data.target

# Using enron1, enron3, and enron5 for training
trainX = X1 + X3 + X5
trainy = numpy.concatenate([y1, y3, y5])

# Using enron2 and enron4 for testing
testX = X2 + X4
testy = numpy.concatenate([y2, y4])


trainDocuments = []
for sen in range(0, len(trainX)):
    # Removing special and single characters
    document = re.sub(r'\W', ' ', str(trainX[sen]))
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

    trainDocuments.append(document)

testDocuments = []
for sen in range(0, len(testX)):
    # Removing special and single characters
    document = re.sub(r'\W', ' ', str(testX[sen]))
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

    testDocuments.append(document)


# Vectorisation
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
trainX = vectorizer.fit_transform(trainDocuments).toarray()

vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
testX = vectorizer.fit_transform(testDocuments).toarray()

# Naive Bayes Results
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(trainX, trainy).predict(testX)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(testy, y_pred))
print(classification_report(testy, y_pred))
print("Naive Bayes: ", accuracy_score(testy, y_pred), "\n")

# Neural Network Results
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlp.fit(trainX, trainy)
predict_test = mlp.predict(testX)
print(confusion_matrix(testy, predict_test))
print(classification_report(testy, predict_test))
print("Neural Network: ", accuracy_score(testy, predict_test), "\n")

# SVM Results
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(trainX, trainy)
predict_test = clf.predict(testX)
print(confusion_matrix(testy, predict_test))
print(classification_report(testy, predict_test))
print("SVM: ", accuracy_score(testy, predict_test))
