# --- 1. Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

# --- 2. Reproducibility ---
import random, os
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- 3. Load Dataset ---
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- 4. Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# --- 5. Define Pipeline / Model ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=500))
])

# --- 6. Cross-Validation (optional) ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = GridSearchCV(clf, {"model__C": [0.1, 1, 10]}, cv=cv, scoring="accuracy")
scores.fit(X_train, y_train)

print("Best Params:", scores.best_params_)
print("CV Score:", scores.best_score_)

# --- 7. Train Final Model ---
best_model = scores.best_estimator_
print(best_model)
best_model.fit(X_train, y_train)

# --- 8. Evaluate on Test Set ---
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ROC AUC (for binary tasks, skip for >2 classes)
try:
    y_proba = best_model.predict_proba(X_test)
    print("ROC AUC:", roc_auc_score(y_test, y_proba, multi_class="ovr"))
except:
    pass

# --- 9. Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.show()

# --- 10. Next Steps / Notes ---
# - Try different models (RandomForest, SVM, HistGradientBoosting)
# - Add preprocessing for categorical features (ColumnTransformer)
# - Tune hyperparameters
