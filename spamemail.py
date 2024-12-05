import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
nltk.download('stopwords')

# Load your dataset (replace with your own dataset file)
# For example, if you have a CSV file named 'spam_emails.csv' with columns 'email_text' and 'label'
df = pd.read_csv('spam_emails.csv')

# Inspect the first few rows of the dataset
print(df.head())

# Preprocessing the dataset:
# Remove any missing values
df = df.dropna()

# Split the dataset into features (X) and target (y)
X = df['email_text']
y = df['label']  # 'label' should be 'spam' or 'ham'

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer for text processing
# This converts email text into a sparse matrix of token counts
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data, then transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model performance:
# Accuracy score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Detailed classification report (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

