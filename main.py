import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import PowerTransformer, KBinsDiscretizer, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from scipy.stats import randint, uniform

# Завантаження даних
url_train = "/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_data.csv"
url_valid = "/kaggle/input/ml-fundamentals-and-applications-2024-10-01/final_proj_test.csv"
train_data = pd.read_csv(url_train)
valid_data = pd.read_csv(url_valid)

# Отримання базової інформації про тренувальний набір даних
train_data_info = train_data.info()
missing_values_summary = train_data.isnull().sum()
categorical_columns = train_data.select_dtypes(include=['object']).columns
unique_categories = {col: train_data[col].nunique() for col in categorical_columns}

# Виведення результатів для аналізу
print("Dataset Information:")
print(train_data_info)
print("\nMissing values per column:")
print(missing_values_summary)
print("\nNumber of unique categories for categorical features:")
print(unique_categories)

# Розподіл класів перед балансуванням
plt.figure(figsize=(10, 5))
sns.countplot(x='y', data=train_data, hue='y', palette='viridis')
plt.title('Class Distribution Before Balancing')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Поділ на ознаки та цільову змінну
X = train_data.drop(columns=['y'])
y = train_data['y']

# Визначення числових та категоріальних ознак і видалення стовпців з єдиним значенням або лише пропусками
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
columns_to_drop = [col for col in categorical_features if train_data[col].nunique() == 1] + X.columns[X.isnull().all()].tolist()

# Видалення стовпців з тренувальних і валідаційних наборів
X.drop(columns=columns_to_drop, inplace=True)
valid_data.drop(columns=columns_to_drop, inplace=True)

# Оновлення списків ознак
numeric_features = numeric_features.difference(columns_to_drop)
categorical_features = categorical_features.difference(columns_to_drop)

# Перетворення категоріальних ознак на тип 'category'
X[categorical_features] = X[categorical_features].astype('category')
valid_data[categorical_features] = valid_data[categorical_features].astype('category')

# Створення препроцессора
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Поділ на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обчислення ваги для класу меншості
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Створення пайплайну
model_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(k_neighbors=5, random_state=42)),
    ('classifier', XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42))
])

# Настройка и навчання моделі для пошуку кращих параметрів
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 4, 5, 6],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.8, 0.9, 1.0],
    'classifier__gamma': [0, 0.1, 0.2],
    'classifier__min_child_weight': [1, 3, 5]
}

random_search = RandomizedSearchCV(model_pipeline, param_distributions=param_dist, n_iter=20, cv=3, random_state=42, n_jobs=-1, scoring='roc_auc')
random_search.fit(X_train, y_train)

# Найкраща модель
best_model = random_search.best_estimator_

# Оцінка моделі
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Використання оптимального порогу для класифікації
optimal_threshold = 0.7175
y_pred_thresh = (y_pred_proba >= optimal_threshold).astype(int)

# Використання оптимального порогу для класифікації
optimal_threshold = 0.7175
y_pred_thresh = (y_pred_proba >= optimal_threshold).astype(int)

# Confusion Matrix з новим порогом
cm = confusion_matrix(y_test, y_pred_thresh)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title("Confusion Matrix (Threshold = 0.58)")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Виведення метрик з новим порогом
print("Accuracy:", accuracy_score(y_test, y_pred_thresh))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_thresh))
print(f'\nROC AUC: {roc_auc_score(y_test, y_pred_proba)}')
print(f'Precision: {precision_score(y_test, y_pred_thresh):.4f}')
print(f'Recall: {recall_score(y_test, y_pred_thresh):.4f}')
print(f'F1 Score: {f1_score(y_test, y_pred_thresh):.4f}')

# Прогнозування для валідаційного набору з оптимальним порогом
valid_predictions_proba = best_model.predict_proba(valid_data)[:, 1]
valid_predictions = (valid_predictions_proba >= optimal_threshold).astype(int)

# Збереження результатів
output = pd.DataFrame({'index': valid_data.index, 'churn': valid_predictions})
output.to_csv('submission.csv', index=False)
print("Прогнози збережені у файлі 'submission.csv'")