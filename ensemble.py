
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

# Base classifiers
lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

dt = DecisionTreeClassifier(random_state=42)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

# Stacking ensemble
stacking = StackingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("knn", knn)
    ],
    final_estimator=LogisticRegression(max_iter=1000)
)

# AdaBoost ensemble
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1, random_state=42
    ),
    n_estimators=50,
    random_state=42
)

# Voting ensemble using majority voting
voting = VotingClassifier(
    estimators=[
        ("lr", lr),
        ("dt", dt),
        ("knn", knn)
    ],
    voting="hard"
)

# Store all models
models = {
    "Logistic Regression": lr,
    "Decision Tree": dt,
    "KNN": knn,
    "Stacking": stacking,
    "AdaBoost": adaboost,
    "Voting": voting
}

# Train and evaluate models
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": round(
            accuracy_score(y_test, y_pred), 4
        ),
        "Precision": round(
            precision_score(y_test, y_pred), 4
        ),
        "Recall": round(
            recall_score(y_test, y_pred), 4
        ),
        "F1-Score": round(
            f1_score(y_test, y_pred), 4
        )
    })

# Display comparison
df = pd.DataFrame(results)

print("\nCLASSIFICATION MODEL COMPARISON")
print(df.to_string(index=False))

# Find best model by accuracy
best = df.loc[df["Accuracy"].idxmax()]

print("\nBest Model:", best["Model"])
print("Best Accuracy:", best["Accuracy"])
