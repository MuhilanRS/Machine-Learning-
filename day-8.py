from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[1], [2], [3], [4], [5], [6], [7], [8]]


y = ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass", "Pass", "Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


model = DecisionTreeClassifier(random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Predicted:", y_pred)
print("Actual:", y_test)
print("Accuracy:", accuracy)


new_student = [[3]]
print("New student:", model.predict(new_student))