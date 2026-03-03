import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================================================
# 1. Load Dataset
# ======================================================
df = pd.read_csv("diabetes.csv")

# Validasi ukuran dataset
print("Ukuran Dataset:", df.shape)

# ======================================================
# 2. Data Cleaning (tanpa chained assignment warning)
# ======================================================
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# ======================================================
# 3. Skema Split (sesuai jurnal)
# ======================================================
splits = {
    "I (90:10)": 0.1,
    "II (80:20)": 0.2,
    "III (70:30)": 0.3,
    "IV (60:40)": 0.4,
    "V (50:50)": 0.5
}

# ======================================================
# 4. Loop Semua Skema
# ======================================================
for name, test_size in splits.items():

    print("\n===================================================")
    print("Skema:", name)
    print("===================================================")

    # Split dulu (hindari data leakage)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )

    # Normalisasi Z-Score (fit hanya pada training)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(C=1.0, kernel='linear', gamma='auto')
    }

    for model_name, model in models.items():

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n--- {model_name} ---")
        print(f"Accuracy  : {acc*100:.2f}%")
        print(f"Precision : {prec*100:.2f}%")
        print(f"Recall    : {rec*100:.2f}%")
        print(f"F1-Score  : {f1*100:.2f}%")
        print("Confusion Matrix:")
        print(cm)