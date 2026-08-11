from sklearn.model_selection import train_test_split

X = [
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [3, 65],
    [7, 85],
    [5, 75],
    [9, 95],
    [1, 50],
    [10, 98]
]

y = [
    "Fail",
    "Pass",
    "Pass",
    "Pass",
    "Fail",
    "Pass",
    "Pass",
    "Pass",
    "Fail",
    "Pass"
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:")
print(X_train)

print("\nTesting data:")
print(X_test)

print("\nTraining labels:")
print(y_train)

print("\nTesting labels:")
print(y_test)