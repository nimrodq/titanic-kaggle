import pandas as pd
import joblib
import os


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from preprocess import preprocess_data



# ==========================
# Load Data
# ==========================

train = pd.read_csv(
    "data/train.csv"
)

test = pd.read_csv(
    "data/test.csv"
)


train, test = preprocess_data(
    train,
    test
)



# ==========================
# Features
# ==========================

features = [

    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Title",

    "FamilySize",
    "FamilySizeGroup",
    "IsAlone",

    "TicketGroupSize",

    "Deck",
    "CabinKnown",

    "WomanChild",

    "FarePerPerson",

    "AgeBand",
    "FareBand"

]


X = train[features]

y = train["Survived"]



# ==========================
# Encoding
# ==========================

X = pd.get_dummies(
    X
)


joblib.dump(
    X.columns.tolist(),
    "outputs/model_columns.pkl"
)



X = X.fillna(
    X.median(numeric_only=True)
)



# ==========================
# Split
# ==========================

X_train, X_valid, y_train, y_valid = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# ==========================
# Models
# ==========================


models = {


    "LogisticRegression":

    LogisticRegression(
        C=0.5,
        max_iter=1000,
        solver="liblinear"
    ),



    "RandomForest":

    RandomForestClassifier(

        n_estimators=500,

        max_depth=6,

        random_state=42

    ),



    "ExtraTrees":

    ExtraTreesClassifier(

        n_estimators=500,

        max_depth=8,

        random_state=42

    ),



    "XGBoost":

    XGBClassifier(

        n_estimators=500,

        learning_rate=0.03,

        max_depth=3,

        random_state=42,

        eval_metric="logloss"

    ),



    "CatBoost":

    CatBoostClassifier(

        iterations=500,

        depth=5,

        learning_rate=0.03,

        verbose=0,

        random_state=42

    )

}



# ==========================
# Train Models
# ==========================


scores = {}


for name, model in models.items():


    model.fit(

        X_train,

        y_train

    )


    pred = model.predict(

        X_valid

    )


    acc = accuracy_score(

        y_valid,

        pred

    )


    scores[name] = acc


    print(

        name,

        acc

    )



# ==========================
# Best Model
# ==========================

best_name = max(

    scores,

    key=scores.get

)


best_model = models[best_name]


print(

    "Best Model:",

    best_name

)



# ==========================
# Retrain Full Dataset
# ==========================


best_model.fit(

    X,

    y

)



# ==========================
# Save
# ==========================


os.makedirs(

    "outputs",

    exist_ok=True

)



joblib.dump(

    best_model,

    "outputs/titanic_model.pkl"

)


print(

    "Model saved!"

)