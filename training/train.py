import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("fruit_spoilage_dataset.csv")
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={"Humid (%)": "Humid", "Light (Fux)": "Light", "CO2 (pmm)": "CO2"})

# Normalize inconsistent label casing ("Bad" vs "BAD")
df["Class"] = df["Class"].str.strip().str.upper()
y = (df["Class"] == "BAD").astype(int)

fruits = sorted(df["Fruit"].unique().tolist())
print("Fruits in dataset:", fruits)
print("Class balance:", df["Class"].value_counts().to_dict())

fruit_onehot = pd.get_dummies(df["Fruit"], prefix="fruit")


def evaluate(feature_df, label, features_desc):
    X_train, X_test, y_train, y_test = train_test_split(feature_df, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    numeric_cols = [c for c in feature_df.columns if c in ("Temp", "Humid", "Light", "CO2")]
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n=== {label} ({features_desc}) ===")
    print("Accuracy:", round(acc, 4))
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["Good", "Bad"]))
    return clf, scaler, numeric_cols, feature_df.columns.tolist()


# Scenario A: full feature set (what the original study used)
full_features = pd.concat([df[["Temp", "Humid", "Light", "CO2"]], fruit_onehot], axis=1)
evaluate(full_features, "Full feature set", "Temp+Humid+Light+CO2+Fruit")

# Scenario B: only what our hardware actually measures (DHT22 = Temp+Humid, plus Fruit selector)
our_features = pd.concat([df[["Temp", "Humid"]], fruit_onehot], axis=1)
clf, scaler, numeric_cols, feature_cols = evaluate(our_features, "Our sensor subset", "Temp+Humid+Fruit only")

# Export the trained logistic regression as plain weights, so inference needs
# no sklearn/pandas/numpy at runtime - just a dot product + sigmoid.
export = {
    "feature_order": feature_cols,
    "numeric_features": numeric_cols,
    "scaler_mean": {c: float(m) for c, m in zip(numeric_cols, scaler.mean_)},
    "scaler_scale": {c: float(s) for c, s in zip(numeric_cols, scaler.scale_)},
    "fruit_categories": fruits,
    "weights": clf.coef_[0].tolist(),
    "bias": float(clf.intercept_[0]),
    "note": "Predicts P(Bad). Standardize Temp/Humid with (x-mean)/scale, one-hot encode fruit in feature_order, dot with weights, add bias, sigmoid.",
}
with open("spoilage_model.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nExported model to spoilage_model.json")
print(json.dumps(export, indent=2))
