import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from preprocess import preprocess_data


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train, test = preprocess_data(
    train,
    test
)



# ==========================
# Feature Cleanup
# ==========================

drop_cols = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin",
    "Survived"
]


X = train[features]
y = train["Survived"]


X = pd.get_dummies(X)

joblib.dump(
    X.columns.tolist(),
    "outputs/model_columns.pkl"
)

X = X.fillna(X.median(numeric_only=True))

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(
    C=0.5,
    max_iter=1000,
    solver="liblinear"
)

model.fit(
    X_train,
    y_train
)


    pred = model.predict(
        X_val
    )


    acc = accuracy_score(
        y_val,
        pred
    )


    scores[name] = acc


    print(
        name,
        acc
    )



# ==========================
# Select Best Model
# ==========================

best_name = max(
    scores,
    key=scores.get
)


best_model = models[
    best_name
]


print(
    "Best Model:",
    best_name
)



# ==========================
# Save Model
# ==========================

best_model.fit(
    X,
    y
)


joblib.dump(
    model,
    "outputs/titanic_model.pkl"
)

print("Model saved!")

predictions = model.predict(X_valid)

accuracy = accuracy_score(
    y_valid,
    predictions
)

print(f"Validation accuracy: {accuracy:.3f}")