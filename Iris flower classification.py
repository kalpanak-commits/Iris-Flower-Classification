Python 3.10.8 (tags/v3.10.8:aaaf517, Oct 11 2022, 16:50:30) [MSC v.1933 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Iris Flower Classification.py
... 
... import pandas as pd
... import numpy as np
... import matplotlib.pyplot as plt
... import seaborn as sns
... 
... from sklearn.datasets import load_iris
... from sklearn.model_selection import train_test_split
... from sklearn.linear_model import LogisticRegression
... from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
... 
... # Load Iris dataset
... iris = load_iris()
... X = pd.DataFrame(iris.data, columns=iris.feature_names)
... y = pd.Series(iris.target, name='species')
... 
... # Map numeric labels to target names
... label_map = dict(enumerate(iris.target_names))
... y_named = y.map(label_map)
... 
... # Combine for easier visualization
... df = X.copy()
... df['species'] = y_named
... 
... # Visualize the data
... sns.pairplot(df, hue='species')
... plt.suptitle('Iris Pairplot', y=1.02)
... plt.show()
... 
... # Split the data
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... # Train a model
... model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
