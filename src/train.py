import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from preprocess import preprocess_data


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

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
y = train['Survived']

X = pd.get_dummies(X)
X = X.fillna(0)

joblib.dump(
    X.columns.tolist(),
    'outputs/model_columns.pkl'
)

models = {
    'Logistic Regression': LogisticRegression(
        C=0.5,
        max_iter=1000,
        solver='liblinear',
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),

    'Extra Trees': ExtraTreesClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    ),

    'XGBoost': XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    ),

    'CatBoost': CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='Accuracy',
        verbose=False,
        random_seed=42
    )
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {}

print('\nModel Comparison')
print('-' * 60)

for name, model in models.items():

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring='accuracy'
    )

    mean_score = scores.mean()
    std_score = scores.std()

    results[name] = mean_score

    print(f'{name:25s} CV: {mean_score:.4f} ± {std_score:.4f}')

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print('\n' + '-' * 60)
print(f'Best Model: {best_model_name}')
print(f'Best CV Score: {results[best_model_name]:.4f}')
print('-' * 60)

best_model.fit(X, y)

joblib.dump(
    best_model,
    'outputs/titanic_model.pkl'
)

joblib.dump(
    best_model_name,
    'outputs/best_model_name.pkl'
)

print(f'\nSaved model: {best_model_name}')
print('Model saved to outputs/titanic_model.pkl')