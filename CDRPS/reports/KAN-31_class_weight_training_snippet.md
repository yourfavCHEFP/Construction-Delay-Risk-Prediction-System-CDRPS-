# KAN-31 Training Snippet (Class Weights)

Use this snippet to train a classifier with class weights derived from the KAN-31 imbalance analysis.

## Recommended Weights

- Class 1: 0.7333
- Class 2: 0.6530
- Class 3: 9.5333

## Example (scikit-learn RandomForestClassifier)

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Example: replace with your prepared dataset and target
# X = df.drop(columns=["Respondant Information.3"])
# y = df["Respondant Information.3"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

class_weights = {
    1: 0.7333,
    2: 0.6530,
    3: 9.5333,
}

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight=class_weights,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, digits=4))
```

## Notes

- Keep resampling (if used) limited to training data only.
- Report macro-F1 and per-class recall to verify minority-class improvement.
- Compare this weighted baseline against an unweighted model for review evidence.
