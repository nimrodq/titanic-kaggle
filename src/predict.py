import pandas as pd
import joblib

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


X_test = test[features]
print(X_test.isnull().sum())

X_test = pd.get_dummies(X_test)


X_test = X_test.fillna(
    X_test.median(numeric_only=True)
)


model_columns = joblib.load(
    "outputs/model_columns.pkl"
)


X_test = X_test.reindex(
    columns=model_columns,
    fill_value=0
)


model = joblib.load(
    "outputs/titanic_model.pkl"
)


predictions = model.predict(X_test).astype(int)

print(pd.Series(predictions).value_counts())
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictions
})


submission.to_csv(
    "outputs/submission.csv",
    index=False
)


print("Submission created!")