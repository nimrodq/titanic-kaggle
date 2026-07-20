import pandas as pd
import numpy as np
import joblib

from preprocess import preprocess_data

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier



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
# Feature Cleanup
# ==========================

drop_cols = [
    "PassengerId",
    "Name",
    "Ticket",
    "Cabin",
    "Survived"
]


X = train.drop(
    columns=drop_cols
)


y = train["Survived"]


test_ids = test["PassengerId"]


X_test = test.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]
)



# ==========================
# Encode Categorical Columns
# ==========================

categorical_cols = X.select_dtypes(
    include=["object"]
).columns


encoders = {}


for col in categorical_cols:

    le = LabelEncoder()


    X[col] = le.fit_transform(
        X[col].astype(str)
    )


    X_test[col] = le.transform(
        X_test[col].astype(str)
    )


    encoders[col] = le



joblib.dump(
    encoders,
    "encoders.pkl"
)



# ==========================
# Clean Data
# ==========================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(
    X.median()
)


X_test = X_test.fillna(
    X.median()
)



print(
    "NaN check:",
    X.isnull().sum().sum()
)


print(
    "Inf check:",
    np.isinf(X).sum().sum()
)



# ==========================
# Split
# ==========================

X_train, X_val, y_train, y_val = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)



# ==========================
# Models
# ==========================

xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    tree_method="hist"
)


cat = CatBoostClassifier(
    verbose=0,
    random_state=42
)


lgbm = LGBMClassifier(
    random_state=42,
    verbosity=-1
)


rf = RandomForestClassifier(
    random_state=42
)



# ==========================
# Hyperparameters
# ==========================

xgb_params = {

    "n_estimators":[300,500,700,1000],

    "max_depth":[2,3,4,5],

    "learning_rate":[0.01,0.03,0.05],

    "subsample":[0.7,0.8,0.9,1.0],

    "colsample_bytree":[0.7,0.8,0.9,1.0],

    "min_child_weight":[1,3,5],

    "gamma":[0,0.1,0.2]

}



cat_params = {

    "iterations":[300,500,800],

    "depth":[4,6,8],

    "learning_rate":[0.01,0.05,0.1],

    "l2_leaf_reg":[1,3,5]

}



lgbm_params = {

    "n_estimators":[100,200,300],

    "learning_rate":[0.01,0.03,0.05],

    "num_leaves":[5,10,15],

    "max_depth":[3,5],

    "min_child_samples":[20,30,50],

    "reg_alpha":[0.1,0.5],

    "reg_lambda":[0.1,0.5]

}



rf_params = {

    "n_estimators":[200,400,600],

    "max_depth":[5,10,20,None],

    "min_samples_split":[2,5,10],

    "min_samples_leaf":[1,2,4]

}



# ==========================
# Tuning Function
# ==========================

def tune(model, params):

    search = RandomizedSearchCV(

        model,

        params,

        n_iter=30,

        cv=5,

        scoring="accuracy",

        random_state=42,

        n_jobs=-1

    )


    search.fit(
        X_train,
        y_train
    )


    print(
        model.__class__.__name__,
        "Best:",
        search.best_score_
    )


    return search.best_estimator_



# ==========================
# Train Models
# ==========================

best_xgb = tune(
    xgb,
    xgb_params
)


best_cat = tune(
    cat,
    cat_params
)


best_lgbm = tune(
    lgbm,
    lgbm_params
)


best_rf = tune(
    rf,
    rf_params
)



# ==========================
# Validation
# ==========================

models = {

    "XGB": best_xgb,

    "CatBoost": best_cat,

    "LightGBM": best_lgbm,

    "RandomForest": best_rf

}


scores = {}



for name, model in models.items():

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
    best_model,
    "model.pkl"
)

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


scores = cross_val_score(
    best_cat,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print(scores.mean())

print(
    "Model saved!"
)