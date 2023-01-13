from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a K-Nearest Neighbors classifier on the training data
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Test the classifier on the test data
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_acc = log_reg.score(X_test, y_test)
print('Logistic Regression accuracy:', log_reg_acc)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_acc = dt.score(X_test, y_test)
print('Decision Tree accuracy:', dt_acc)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)
print('Random Forest accuracy:', rf_acc)
