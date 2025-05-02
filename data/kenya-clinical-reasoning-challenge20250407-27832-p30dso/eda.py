import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import missingno as msno

df = pd.read_csv('/Users/naomiamadi/Downloads/Documents/Zindi Clinical Reasoning Challenge/train.csv')

pd.set_option('display.max_columns', None)  # show all columns

 # Quick overview of the dataset
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nInfo:")
df.info()
print("\nSummary Statistics:")
print(df.describe(include='all'))

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("Missing values:\n", missing_values)

msno.matrix(df)
plt.show()

# Check if 'target' column exists
if 'target' in df.columns:
    print(df['target'].value_counts())
    sns.countplot(data=df, x='target')
    plt.title('Target Variable Distribution')
    plt.show()

# Separate categorical and numerical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical Columns: {categorical_cols}")
print(f"Numerical Columns: {numerical_cols}")

# Histograms for numerical features
df[numerical_cols].hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()

# Bar plots for categorical features
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

# Correlation matrix to show correlation between numerical variables (bivariate analysis)
corr = df[numerical_cols].corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Outlier Detection
# Boxplots for numerical variables
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=col)
    plt.title(f'Boxplot of {col}')
    plt.show()

# Feature-Target Relationships
# If 'target' exists
if 'target' in df.columns:
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x='target', y=col, data=df)
        plt.title(f'{col} vs Target')
        plt.show()

# Categorical Features vs Target
# If 'target' exists
if 'target' in df.columns:
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, hue='target', data=df)
        plt.title(f'{col} vs Target')
        plt.xticks(rotation=45)
        plt.show()

# Summary Observations
# Display some basic summary
print("Summary Observations:")
print("- Dataset shape:", df.shape)
print("- Number of missing values:", missing_values.sum())
print("- Numerical features:", len(numerical_cols))
print("- Categorical features:", len(categorical_cols))
print("- Target distribution (if exists):")
if 'target' in df.columns:
    print(df['target'].value_counts(normalize=True))

# Submission Preparation

from sklearn.preprocessing import LabelEncoder

# Load test dataset
test_df = pd.read_csv('/Users/naomiamadi/Downloads/Documents/Zindi Clinical Reasoning Challenge/test.csv')

# Preprocess test data using same strategy as training data
for col in numerical_cols:
    test_df[col] = test_df[col].fillna(df[col].median())

for col in categorical_cols:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    test_df[col] = le.transform(test_df[col].fillna(df[col].mode()[0]).astype(str))

# Optional: Aggregate features for test set
if len(numerical_cols) >= 2:
    test_df['feature_sum'] = test_df[numerical_cols].sum(axis='columns')
    test_df['feature_mean'] = test_df[numerical_cols].mean(axis='columns')

# Prepare training data for model
X = df.drop(columns=['target'])
y = df['target']

# Ensure the features match between train and test
X = X[test_df.columns.drop('ID')]

# Use only the columns present in training data for prediction
X_test = test_df[X.columns]

# Train final model on entire training set
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

rf = RandomForestClassifier(n_estimators=100, random_state=42)
logreg = LogisticRegression(max_iter=1000)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
gbm = GradientBoostingClassifier()

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('logreg', logreg),
        ('xgb', xgb_model),
        ('gbm', gbm)
    ],
    voting='soft'
)

ensemble_model.fit(X, y)

# Predict and prepare submission
test_df['target'] = ensemble_model.predict(X_test)

# Save to submission file
submission = test_df[['ID', 'target']]  # Ensure 'ID' is the correct identifier column
submission.to_csv('/Users/naomiamadi/Downloads/Documents/Zindi Clinical Reasoning Challenge/submission.csv', index=False)
print('✅ Submission file generated: submission.csv')