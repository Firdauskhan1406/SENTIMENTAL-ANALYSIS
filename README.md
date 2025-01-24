# SENTIMENTAL-ANALYSIS

"COMAPNY" : CODTECH IT SOLUTIONS

"NAME" : FIRDAUS KHAN

"INTERN ID" : CT08NJP

"DOMAIN": DATA ANALYTICS

"DURATION: 4 WEEKS

"MENTOR" : NEELA SANTOSH



Explanation of the Code
Importing Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
pandas: Used for data manipulation and analysis.
numpy: Useful for numerical computations (though not explicitly used here).
train_test_split: Splits the dataset into training and testing sets.
CountVectorizer: Converts text data into numerical feature vectors.
MultinomialNB: A Naive Bayes algorithm suitable for text classification.
accuracy_score, classification_report, confusion_matrix: Tools to evaluate model performance.
Loading the Dataset
python
Copy
Edit
data = pd.read_csv('sentiment_dataset.csv')
print(data.head())
pd.read_csv: Reads a CSV file containing text data and sentiment labels.
data.head(): Displays the first few rows to verify the dataset's structure.
Splitting Features and Labels
python
Copy
Edit
X = data['text']
y = data['sentiment']
X contains the text data (features).
y contains the sentiment labels (target variable).
Splitting Training and Testing Data
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Splits the data into 80% training and 20% testing.
random_state ensures consistent results across runs.
Text Vectorization
python
Copy
Edit
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
CountVectorizer: Transforms text data into numerical feature vectors (bag-of-words representation).
fit_transform: Fits the vectorizer to training data and transforms it.
transform: Transforms the test data using the same vectorizer.
Model Training
python
Copy
Edit
model = MultinomialNB()
model.fit(X_train_vect, y_train)
Initializes a Naive Bayes classifier and trains it on the vectorized training data.
Predictions
python
Copy
Edit
y_pred = model.predict(X_test_vect)
Uses the trained model to predict sentiments on the test data.
Model Evaluation
python
Copy
Edit
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
Calculates and prints:
Accuracy: Proportion of correctly classified samples.
Classification Report: Precision, recall, and F1-score for each class.
Confusion Matrix: Summarizes prediction results against actual values.
Expected Outputs
Dataset Preview: Displays the first rows of the dataset.
Accuracy: Numeric value showing the percentage of correct predictions.
Classification Report: Detailed performance metrics for each class.
Confusion Matrix: A matrix showing true vs. predicted classifications.
Step-by-Step Detailed Explanation of the Sentiment Analysis Code
1. Importing Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
pandas:

This library is used to handle data in tabular form (like CSV files).
Functions such as read_csv() help load data, and methods like head() display the first few rows for inspection.
numpy:

A powerful library for numerical computations and array operations. While not explicitly used in this code, it can be helpful when performing advanced numerical transformations, such as converting predictions into arrays or working with probabilities.
train_test_split:

A utility function from scikit-learn that divides data into training and testing subsets.
This ensures that we can train our model on one part of the data and evaluate its performance on another unseen part.
CountVectorizer:

A feature extraction tool for text data.
It converts raw text into a matrix of word counts, known as a bag-of-words model.
For example, the text "I love programming" becomes a sparse matrix indicating word frequencies.
MultinomialNB:

This is the Multinomial Naive Bayes classifier, specifically designed for discrete data, such as word frequencies.
It assumes conditional independence between features given the class.
accuracy_score, classification_report, confusion_matrix:

These tools evaluate the performance of the model:
accuracy_score: The proportion of correctly classified samples.
classification_report: Shows detailed metrics like precision (how many predicted labels are correct), recall (how many actual labels were predicted correctly), and F1-score (harmonic mean of precision and recall).
confusion_matrix: A table that compares actual vs. predicted labels.
2. Loading the Dataset
python
Copy
Edit
data = pd.read_csv('sentiment_dataset.csv')
print(data.head())
The read_csv() function loads the data from a CSV file into a DataFrame.
Dataset structure: The dataset should typically contain at least two columns:
text: The text samples we aim to classify.
sentiment: The labels corresponding to the sentiment of each text (e.g., positive, negative, neutral).
data.head(): Displays the first five rows to verify the data was loaded correctly.
Example output:
kotlin
Copy
Edit
   text                            sentiment
0  I love this product!          positive
1  This is terrible...             negative
2  It's okay, nothing special   neutral
3. Separating Features and Labels
python
Copy
Edit
X = data['text']
y = data['sentiment']
X (features):

Represents the input data (text column).
The raw text samples that need to be processed into numerical form.
y (labels):

Represents the target variable (sentiment column).
Each entry corresponds to the sentiment of the corresponding text.
4. Splitting Training and Testing Data
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Purpose:

Divides the dataset into training (80%) and testing (20%) subsets.
This ensures the model is trained on one part and tested on an unseen portion.
Parameters:

test_size=0.2: Reserves 20% of the data for testing.
random_state=42: Ensures consistent splitting across different runs by fixing the random seed.
5. Converting Text to Numerical Data
python
Copy
Edit
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)
Purpose:

Text data is inherently non-numerical, so it must be converted into numerical form.
The CountVectorizer transforms text into a bag-of-words representation:
Each document is represented as a vector indicating the frequency of each word.
Key Steps:

fit_transform():
Learns the vocabulary from the training data and converts it into a sparse matrix.
Example:
Vocabulary: {"love": 0, "this": 1, "product": 2, "terrible": 3}
Text "I love this product" becomes [1, 1, 1, 0].
transform():
Converts the test data into numerical vectors using the vocabulary learned from the training set.
6. Training the Model
python
Copy
Edit
model = MultinomialNB()
model.fit(X_train_vect, y_train)
Purpose:
Trains the Multinomial Naive Bayes model using the vectorized training data (X_train_vect) and the corresponding labels (y_train).
How it works:
The model calculates the likelihood of each word given a class (e.g., positive).
Based on Bayes' Theorem, it predicts the class for new samples.
7. Making Predictions
python
Copy
Edit
y_pred = model.predict(X_test_vect)
The trained model uses the test data (X_test_vect) to predict sentiment labels.

8. Evaluating the Model
python
Copy
Edit
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
accuracy_score:
Provides a single metric showing the percentage of correct predictions.
classification_report:
Displays detailed metrics:
Precision: Correctly predicted positive samples / Total predicted positive samples.
Recall: Correctly predicted positive samples / Total actual positive samples.
F1-Score: Balances precision and recall.
confusion_matrix:
Shows how well the model performed across all classes:
Rows: Actual labels.
Columns: Predicted labels.
Additional Context and Insights
1. Why Naive Bayes?
Naive Bayes is fast, simple, and effective for text classification.
It works well even for large datasets and sparse data (like text).
2. Limitations of Naive Bayes
Assumes word independence, which might not capture context in sentences.
Struggles with out-of-vocabulary words (e.g., rare or unseen words in the test data).
3. Improvements
Replace Naive Bayes with more advanced models like Logistic Regression or Support Vector Machines.
Use TF-IDF (Term Frequency-Inverse Document Frequency) instead of simple bag-of-words to give importance to rare but significant words.
Conclusion
This sentiment analysis project effectively demonstrates a basic pipeline for text classification. It leverages foundational concepts like bag-of-words, Naive Bayes, and performance metrics. While the results may be effective, improvements like advanced feature extraction (e.g., TF-IDF) or more sophisticated models (e.g., deep learning) can further enhance performance.
