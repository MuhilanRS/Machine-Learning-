from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X = [
    [1000, 2, 10],
    [1200, 2, 8],
    [1500, 3, 5],
    [1800, 3, 4],
    [2000, 4, 3],
    [2200, 4, 2],
    [2500, 4, 1],
    [2800, 5, 1],
    [3000, 5, 2],
    [3500, 5, 1]
]

y = [35, 40, 55, 65, 75, 85, 100, 115, 125, 145]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual prices:", y_test)
print("Predicted prices:", y_pred)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))