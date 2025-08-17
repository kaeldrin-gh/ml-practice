# Exercise: Iris Classification with Random Forest
# Objective : Train a Random Forest Classifier on the Iris dataset.
# Evaluate using accuracy, precision, recall, F1-score.
# Plot feature importances to see which features drive predictions.

# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Reproducibility ---
SEED = 42
np.random.seed(SEED)

# --- Load Dataset ---
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# --- Define Pipeline ---
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=SEED))
])

# --- Hyperparameter Grid ---
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [None, 3, 5],
    "model__min_samples_split": [2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(clf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best CV Score:", grid_search.best_score_)

# --- Evaluate on Test Set ---
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(cmap="Blues", values_format="d")
plt.show()

# --- Feature Importances ---
importances = best_model.named_steps["model"].feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importances (Random Forest)")
plt.show()
