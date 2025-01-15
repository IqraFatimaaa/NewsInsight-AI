# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:07:24 2025

@author: PMYLS
"""
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('stopwords')

# Printing the stopwords in English
print(stopwords.words('english'))

# Load the dataset
news_dataset = pd.read_csv(r'C:/Users/PMLS/Desktop/FakeNewsNet.csv')

# Check the shape and top rows of the dataset
print(news_dataset.shape)
print(news_dataset.head())

# Check for missing values
print(news_dataset.isnull().sum())

# Fill missing values
news_dataset = news_dataset.fillna('')

# Merge source_domain and title as content
news_dataset['content'] = news_dataset['source_domain'] + ' ' + news_dataset['title']
print(news_dataset['content'].head())

# Separating the data and label
X = news_dataset['content'].values
Y = news_dataset['real'].values
print(X)
print(Y)

# Stemming function
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming to the content column
news_dataset['content'] = news_dataset['content'].apply(stemming)
print(news_dataset['content'].head())

# Converting the textual data to numerical data using TfidfVectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print(X)

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data:', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data:', test_data_accuracy)

# Predicting for a new test instance
X_new = X_test[3]
prediction = model.predict(X_new)

# Print the prediction result
if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')

# Print the actual label
print('Actual label:', Y_test[3])
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(news_dataset['content'])

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, news_dataset['real'], test_size=0.3, random_state=42)

# Step 3: Model Building (Logistic Regression or Random Forest)
# Using Logistic Regression for simplicity
model = LogisticRegression(max_iter=1000)
# Alternatively, you can use a RandomForestClassifier if you prefer:
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Model Training
model.fit(X_train, y_train)

# Step 5: Model Evaluation

# Predict on test data
y_pred = model.predict(X_test)

# Print Accuracy and Classification Report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Confusion Matrix Visualization (Optional but useful)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
