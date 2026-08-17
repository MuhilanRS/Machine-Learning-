from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X = [[2], [4], [6], [8], [10], [12], [14], [16]]
y = [0, 0, 0, 1, 1, 1, 1, 1]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Accuracy:", accuracy)