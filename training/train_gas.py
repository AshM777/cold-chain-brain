import glob
import json
import re

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load every single-fruit "<Fruit> D<day>.csv" file (skip mixed-fruit combos and
# the non-fruit "Mushroom" files). Day is 1 (freshest, just sampled) through 5
# (5 days later). There's no independent freshness label in this dataset, so we
# define one ourselves: Day >= 4 => "Bad". This is a modeling assumption, not a
# verified ground truth from the source study - flagged here and in main.py.
rows = []
for path in glob.glob("AllSmaples-Report/*.csv"):
    m = re.match(r"^([A-Za-z ]+?)\s*D(\d)\.csv$", path.split("/")[-1])
    if not m:
        continue
    fruit = m.group(1).strip().title()
    if fruit == "Mushroom":
        continue
    day = int(m.group(2))
    df = pd.read_csv(path, usecols=["MQ3", "MQ135"])
    df["Fruit"] = fruit
    df["Day"] = day
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
data["Bad"] = (data["Day"] >= 4).astype(int)

print("Total rows:", len(data))
print("Fruits:", sorted(data["Fruit"].unique().tolist()))
print("Class balance:", data["Bad"].value_counts().to_dict())
print("MQ3 range:", data["MQ3"].min(), data["MQ3"].max())
print("MQ135 range:", data["MQ135"].min(), data["MQ135"].max())

fruit_onehot = pd.get_dummies(data["Fruit"], prefix="fruit")
X = pd.concat([data[["MQ3", "MQ135"]], fruit_onehot], axis=1)
y = data["Bad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
numeric_cols = ["MQ3", "MQ135"]
X_train = X_train.copy()
X_test = X_test.copy()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, preds), 4))
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, target_names=["Good", "Bad"]))

fruits_sorted = sorted(data["Fruit"].unique().tolist())
export = {
    "feature_order": ["MQ3", "MQ135"] + [f"fruit_{f}" for f in fruits_sorted],
    "numeric_features": numeric_cols,
    "scaler_mean": {c: float(m) for c, m in zip(numeric_cols, scaler.mean_)},
    "scaler_scale": {c: float(s) for c, s in zip(numeric_cols, scaler.scale_)},
    "fruit_categories": fruits_sorted,
    "weights": clf.coef_[0].tolist(),
    "bias": float(clf.intercept_[0]),
    "label_definition": "Bad = 1 if the sample was taken on day 4 or 5 of a 5-day monitoring window, else Good. Not an independently verified spoilage label.",
    "source": "Food Freshness Electronic Nose Dataset, Zenodo doi:10.5281/zenodo.17285312, CC BY 4.0",
    "caveat": "Trained on this study's own MQ3/MQ135 units; MQ3 range (53-941) roughly matches a 0-1023 ADC like ours, but MQ135 range (434-2975) exceeds our 0-1023 ADC range, meaning their sensor/circuit was scaled differently. Treat gas-based predictions as directional, not precisely calibrated.",
}
with open("gas_spoilage_model.json", "w") as f:
    json.dump(export, f, indent=2)

print("\nExported to gas_spoilage_model.json")
print(json.dumps({k: v for k, v in export.items() if k not in ("weights",)}, indent=2))
print("weights:", export["weights"])
