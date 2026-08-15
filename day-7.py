
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]


y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Test Data:", X_test)
print("Actual Results:", y_test)
print("Predicted Results:", y_pred)
print("Accuracy:", accuracy)

new_student = [[6]]
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("New Student: Pass")
else:
    print("New Student: Fail")