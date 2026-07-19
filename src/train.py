import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import preprocess_data


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train, test = preprocess_data(train, test)

features = [
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "FamilySize",
    "IsAlone",
    "Embarked",
    "Title"
]

X = train[features]
y = train["Survived"]


X = pd.get_dummies(X)

X = X.fillna(X.median(numeric_only=True))

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
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