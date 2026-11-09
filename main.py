import os
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
RAW_DIR = os.path.join('data', 'raw')
SUB_DIR = os.path.join('data', 'submissions')
REP_DIR = 'reports'
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(REP_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(RAW_DIR, 'test.csv'))

print('Train:', train_df.shape, '| Test:', test_df.shape)
print(train_df['Status'].value_counts(), '\n')


def create_features(df):
    df = df.copy()

    df['Age_years'] = df['Age'] / 365.25
    df = df.drop(columns=['Age'])

    df['has_complications'] = ((df['Ascites'].isin(['Y', 'S'])) |
                               (df['Edema'].isin(['Y', 'S']))).astype(int)
    df['bili_albumin_ratio'] = df['Bilirubin'] / (df['Albumin'] + 1e-6)
    df['severity_index'] = df['Stage'] * (1 + df['has_complications'])
    df['high_bilirubin'] = (df['Bilirubin'] > 2.0).astype(int)
    df['low_albumin'] = (df['Albumin'] < 3.5).astype(int)
    df['low_platelets'] = (df['Platelets'] < 150).astype(int)

    bins = [0, 40, 50, 60, 70, 120]
    labels = ['young', 'middle', 'senior', 'elderly', 'old']
    df['age_group'] = pd.cut(df['Age_years'], bins=bins, labels=labels, include_lowest=True)
    df['age_group'] = df['age_group'].astype(object)

    df['bili_stage_interaction'] = df['Bilirubin'] * df['Stage']
    df['albumin_platelets_interaction'] = df['Albumin'] * df['Platelets']
    df['log_bilirubin'] = np.log1p(df['Bilirubin'].clip(lower=0))
    df['log_alk_phos'] = np.log1p(df['Alk_Phos'].clip(lower=0))
    df['log_copper'] = np.log1p(df['Copper'].clip(lower=0))
    df['all_symptoms'] = ((df['Ascites'] == 'Y') & (df['Edema'] == 'Y') &
                          (df['Spiders'] == 'Y') & (df['Hepatomegaly'] == 'Y')).astype(int)
    return df


train_df = create_features(train_df)
test_df = create_features(test_df)

print('age_group после исправления:')
print(train_df['age_group'].value_counts(dropna=False), '\n')

X = train_df.drop(columns=['id', 'Status'])
y = train_df['Status']
X_test = test_df.drop(columns=['id'])
test_ids = test_df['id'].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

categorical_features = ['Drug', 'Sex', 'Ascites', 'Edema', 'Spiders',
                        'Hepatomegaly', 'age_group']
numeric_features = [c for c in X.columns if c not in categorical_features]


def make_preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}
probas_val = {}
probas_test = {}
models = {}


def evaluate(name, pipeline, y_tr, y_v, encoded=False, le=None):
    scores = cross_val_score(pipeline, X_train, y_tr, cv=cv, scoring='f1_macro', n_jobs=1)
    pipeline.fit(X_train, y_tr)

    pred_enc = pipeline.predict(X_val)
    proba = pipeline.predict_proba(X_val)
    pred = le.inverse_transform(pred_enc) if encoded else pred_enc

    results[name] = {
        'CV F1-macro': scores.mean(),
        'CV std': scores.std(),
        'Val Accuracy': accuracy_score(y_val, pred),
        'Val F1-macro': f1_score(y_val, pred, average='macro'),
        'Val LogLoss': log_loss(y_val, proba, labels=le.classes_ if encoded else None)
    }
    probas_val[name] = proba
    probas_test[name] = pipeline.predict_proba(X_test)
    models[name] = pipeline

    print('=' * 60)
    print(name)
    print('=' * 60)
    print(f'CV F1-macro: {scores.mean():.4f} (+/- {scores.std():.4f})')
    print(f'Val Accuracy: {results[name]["Val Accuracy"]:.4f} | '
          f'Val F1-macro: {results[name]["Val F1-macro"]:.4f} | '
          f'Val LogLoss: {results[name]["Val LogLoss"]:.4f}')
    print(classification_report(y_val, pred, zero_division=0))


lr_pipeline = Pipeline([
    ('preprocessor', make_preprocessor()),
    ('classifier', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                      class_weight='balanced'))
])
evaluate('Logistic Regression', lr_pipeline, y_train, y_val)

rf_pipeline = Pipeline([
    ('preprocessor', make_preprocessor()),
    ('classifier', RandomForestClassifier(n_estimators=300, max_depth=10,
                                          min_samples_split=5, min_samples_leaf=2,
                                          class_weight='balanced',
                                          random_state=RANDOM_STATE, n_jobs=-1))
])
evaluate('Random Forest', rf_pipeline, y_train, y_val)

le = LabelEncoder()
le.fit(y)
y_train_enc = le.transform(y_train)

xgb_pipeline = Pipeline([
    ('preprocessor', make_preprocessor()),
    ('classifier', xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, objective='multi:softprob',
                                     num_class=len(le.classes_), eval_metric='mlogloss',
                                     random_state=RANDOM_STATE, n_jobs=-1))
])
evaluate('XGBoost', xgb_pipeline, y_train_enc, y_val, encoded=True, le=le)

comparison = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
comparison = comparison.sort_values('Val LogLoss').reset_index(drop=True)
print('\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===')
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(REP_DIR, 'model_comparison.csv'), index=False)

print('\n=== ВАЖНОСТЬ ПРИЗНАКОВ ===')
for name in ['Random Forest', 'XGBoost']:
    fitted_pre = models[name].named_steps['preprocessor']
    feat_names = fitted_pre.get_feature_names_out()
    feat_names = [f.split('__', 1)[-1] for f in feat_names]
    imp = models[name].named_steps['classifier'].feature_importances_
    fi = (pd.DataFrame({'Feature': feat_names, 'Importance': imp})
          .sort_values('Importance', ascending=False).reset_index(drop=True))
    fi.to_csv(os.path.join(REP_DIR, f'feature_importance_{name.replace(" ", "_").lower()}.csv'),
              index=False)
    print(f'\nTop-15: {name}')
    print(fi.head(15).to_string(index=False))

print('\n=== ФОРМИРОВАНИЕ САБМИШНОВ ===')
class_order = list(le.classes_)
assert class_order == ['C', 'CL', 'D'], class_order

for name, proba in probas_test.items():
    sub = pd.DataFrame({'id': test_ids})
    for i, cls in enumerate(class_order):
        sub[f'Status_{cls}'] = proba[:, i]
    fname = name.replace(' ', '_').lower()
    path = os.path.join(SUB_DIR, f'submission_{fname}.csv')
    sub.to_csv(path, index=False)
    print(f'{path} | shape={sub.shape} | sum_probs≈{sub.iloc[:, 1:].sum(axis=1).mean():.4f}')

best_model = comparison.loc[0, 'Model']
print(f'\nЛучшая модель по LogLoss: {best_model}')
print('Готово.')