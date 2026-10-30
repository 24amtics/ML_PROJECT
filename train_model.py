"""Train a demo classifier using scikit-learn's Wisconsin Diagnostic Breast Cancer dataset."""
from pathlib import Path
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

OUT = Path(__file__).parent / "breast_cancer_model.joblib"

data = load_breast_cancer()
X, y = data.data, data.target  # target 0 = malignant, 1 = benign
feature_names = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry",
    "mean_fractal_dimension", "error_radius", "error_texture", "error_perimeter",
    "error_area", "error_smoothness", "error_compactness", "error_concavity",
    "error_concave_points", "error_symmetry", "error_fractal_dimension",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
    "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry",
    "worst_fractal_dimension"
]
# Human-readable labels for the website
pretty = {
    "radius": "Radius", "texture": "Texture", "perimeter": "Perimeter", "area": "Area",
    "smoothness": "Smoothness", "compactness": "Compactness", "concavity": "Concavity",
    "concave_points": "Concave points", "symmetry": "Symmetry", "fractal_dimension": "Fractal dimension"
}
descriptions = {}
for name in feature_names:
    parts = name.split("_", 1)
    group, metric = parts[0], parts[1]
    group_label = {"mean": "Mean", "error": "Standard error", "worst": "Worst"}.get(group, group.title())
    descriptions[name] = f"{group_label} {pretty.get(metric, metric.replace('_', ' ').title())}"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000, random_state=42))
])
model.fit(X_train, y_train)
preds = model.predict(X_test)
print("Hold-out accuracy:", round(accuracy_score(y_test, preds) * 100, 2), "%")
print("Confusion matrix (rows=true, columns=predicted):\n", confusion_matrix(y_test, preds))
print(classification_report(y_test, preds, target_names=data.target_names))

bundle = {
    "model": model,
    "feature_names": feature_names,
    "feature_descriptions": descriptions,
    "example_values": X.mean(axis=0).tolist()
}
joblib.dump(bundle, OUT)
print(f"Saved model to {OUT}")
