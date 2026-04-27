import pandas as pd
import pickle

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\churn_project\\data\\telco_churn.csv')

# Drop unnecessary columns
df = df.drop(['Unnamed: 0', 'customerID'], axis=1)

# =========================
# FIX NUMERIC COLUMNS
# =========================

# TotalCharges (convert to numeric)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# SeniorCitizen (handle all formats)
df['SeniorCitizen'] = df['SeniorCitizen'].astype(str).str.strip()
df['SeniorCitizen'] = df['SeniorCitizen'].replace({
    'Yes': 1, 'No': 0,
    'True': 1, 'False': 0,
    '1': 1, '0': 0
})
df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)

# =========================
# CLEAN TARGET (VERY IMPORTANT)
# =========================

print("Unique Churn BEFORE:", df['Churn'].unique())

df['Churn'] = df['Churn'].astype(str).str.strip()
df['Churn'] = df['Churn'].replace({
    'Yes': 1, 'No': 0,
    'True': 1, 'False': 0,
    '1': 1, '0': 0
})

df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')
df = df.dropna(subset=['Churn'])
df['Churn'] = df['Churn'].astype(int)

print("Unique Churn AFTER:", df['Churn'].unique())

# =========================
# HANDLE MISSING VALUES
# =========================

# Select categorical columns EXCLUDING target
cat_cols = df.select_dtypes(include=['object', 'string']).columns
cat_cols = cat_cols.drop('Churn', errors='ignore')

# Fill categorical missing
for col in cat_cols:
    df[col] = df[col].fillna('No')

# Drop any remaining missing values
df = df.dropna()

# =========================
# ENCODING
# =========================

df = pd.get_dummies(df, drop_first=True)

# =========================
# SPLIT DATA
# =========================

from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

print("FINAL Churn null values:", y.isnull().sum())
print("FINAL dataset shape:", df.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# TRAIN MODEL
# =========================

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

# =========================
# SAVE MODEL
# =========================

pickle.dump(model, open('../churn_model.pkl', 'wb'))
pickle.dump(X.columns, open('../columns.pkl', 'wb'))

print("✅ Model trained and saved successfully!")