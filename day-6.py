from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


X = [[1], [2], [3], [4], [5],
     [6], [7], [8], [9], [10]]

y = [0, 0, 0, 0, 0,
     1, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print("Testing data:", X_test)
print("Actual values:", y_test)
print("Predicted values:", y_pred)
print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")