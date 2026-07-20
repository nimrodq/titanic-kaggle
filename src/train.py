import pandas as pd
import joblib
import os


from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from preprocess import preprocess_data

import matplotlib.pyplot as plt
import seaborn as sns



# ==========================
# Create Directories
# ==========================

os.makedirs(
    "outputs",
    exist_ok=True
)

os.makedirs(
    "assets",
    exist_ok=True
)



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

    "FamilySizeGroup",

    "TicketGroupSize",

    "CabinKnown",

    "WomanChild"

]


X = train[features]

y = train["Survived"]



# ==========================
# Encoding
# ==========================

X = pd.get_dummies(
    X,
    drop_first=True
)


X = X.fillna(
    X.median(numeric_only=True)
)


joblib.dump(
    X.columns.tolist(),
    "outputs/model_columns.pkl"
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
# Stratified K-Fold
# ==========================

skf = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)



# ==========================
# Models
# ==========================

models = {


    "LogisticRegression":

    LogisticRegression(

        C=0.1,

        max_iter=3000,

        solver="liblinear"

    ),



    "RandomForest":

    RandomForestClassifier(

        n_estimators=1000,

        max_depth=6,

        min_samples_split=4,

        min_samples_leaf=2,

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

        n_estimators=300,

        learning_rate=0.03,

        max_depth=2,

        min_child_weight=3,

        subsample=0.8,

        colsample_bytree=0.8,

        reg_lambda=5,

        random_state=42,

        eval_metric="logloss"

    ),



    "CatBoost":

    CatBoostClassifier(

        iterations=1000,

        depth=4,

        learning_rate=0.02,

        l2_leaf_reg=5,

        random_state=42,

        verbose=0

    )

}



# ==========================
# Training
# ==========================

scores = {}

validation_predictions = {}


threshold = 0.45



for name, model in models.items():


    print(
        f"\nTraining {name}"
    )


    cv_scores = cross_val_score(

        model,

        X,

        y,

        cv=skf,

        scoring="accuracy"

    )


    mean_score = cv_scores.mean()

    std_score = cv_scores.std()


    scores[name] = mean_score



    print(
        f"{name}: {mean_score:.4f} (+/- {std_score:.4f})"
    )



    # Validation model

    model.fit(

        X_train,

        y_train

    )


    prob = model.predict_proba(

        X_valid

    )[:,1]


    pred = (

        prob > threshold

    ).astype(int)



    validation_predictions[name] = pred




# ==========================
# Save Scores
# ==========================

pd.DataFrame(

    scores.items(),

    columns=[

        "Model",

        "Accuracy"

    ]

).sort_values(

    "Accuracy",

    ascending=False

).to_csv(

    "outputs/model_scores.csv",

    index=False

)



# ==========================
# Best Model
# ==========================

best_name = max(

    scores,

    key=scores.get

)


print(

    "\nBest Model:",

    best_name

)



best_model = models[best_name]


best_predictions = validation_predictions[best_name]



# ==========================
# Evaluation
# ==========================

print(

    classification_report(

        y_valid,

        best_predictions

    )

)



# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(

    y_valid,

    best_predictions

)


plt.figure(

    figsize=(5,4)

)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=[

        "Not Survived",

        "Survived"

    ],

    yticklabels=[

        "Not Survived",

        "Survived"

    ]

)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    f"Confusion Matrix - {best_name}"
)


plt.tight_layout()


plt.savefig(

    "assets/confusion_matrix.png",

    dpi=300

)


plt.close()



# ==========================
# Feature Importance Model
# ==========================

importance_model = models[best_name]


importance_model.fit(

    X_train,

    y_train

)



if isinstance(

    importance_model,

    CatBoostClassifier

):

    importance = importance_model.get_feature_importance()



elif hasattr(

    importance_model,

    "feature_importances_"

):

    importance = importance_model.feature_importances_



elif hasattr(

    importance_model,

    "coef_"

):

    importance = abs(

        importance_model.coef_[0]

    )



else:

    result = permutation_importance(

        importance_model,

        X_valid,

        y_valid,

        n_repeats=10,

        random_state=42

    )


    importance = result.importances_mean




feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": importance

})


feature_importance = feature_importance.sort_values(

    "Importance",

    ascending=False

)



print(

    feature_importance.head(15)

)



feature_importance.to_csv(

    "outputs/feature_importance.csv",

    index=False

)



# ==========================
# Feature Importance Plot
# ==========================

top_features = feature_importance.head(15)



plt.figure(

    figsize=(8,6)

)


sns.barplot(

    data=top_features,

    x="Importance",

    y="Feature"

)


plt.title(

    f"Feature Importance - {best_name}"

)


plt.tight_layout()



plt.savefig(

    "assets/feature_importance.png",

    dpi=300

)


plt.close()



# ==========================
# Retrain Full Dataset
# ==========================

best_model.fit(

    X,

    y

)



# ==========================
# Save Model
# ==========================

joblib.dump(

    best_model,

    "outputs/titanic_model.pkl"

)


joblib.dump(

    best_name,

    "outputs/best_model_name.pkl"

)


joblib.dump(

    threshold,

    "outputs/threshold.pkl"

)



print(

    "\nModel saved!"

)

print(

    "Generated assets:"

)

print(

    "- assets/confusion_matrix.png"

)

print(

    "- assets/feature_importance.png"

)