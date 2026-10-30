from flask import Flask, render_template, request
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path(__file__).parent / "breast_cancer_model.joblib"
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
feature_names = model_bundle["feature_names"]
feature_descriptions = model_bundle["feature_descriptions"]

# Example values are dataset averages, provided only to help demonstrate the form.
example_values = model_bundle["example_values"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    values = {name: str(example_values[i]) for i, name in enumerate(feature_names)}

    if request.method == "POST":
        try:
            submitted = {}
            for name in feature_names:
                raw = request.form.get(name, "").strip()
                if not raw:
                    raise ValueError(f"Please enter a value for {feature_descriptions[name]}.")
                value = float(raw)
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{feature_descriptions[name]} must be a valid non-negative number.")
                submitted[name] = value
            values = {name: str(submitted[name]) for name in feature_names}
            row = np.array([[submitted[name] for name in feature_names]], dtype=float)
            prediction = int(model.predict(row)[0])
            probabilities = model.predict_proba(row)[0]
            # sklearn dataset target: 0 = malignant, 1 = benign
            malignant_probability = float(probabilities[0]) * 100
            benign_probability = float(probabilities[1]) * 100
            result = {
                "label": "Benign-like pattern" if prediction == 1 else "Malignant-like pattern",
                "kind": "benign" if prediction == 1 else "malignant",
                "malignant_probability": round(malignant_probability, 1),
                "benign_probability": round(benign_probability, 1),
            }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "Something went wrong while analyzing the measurements. Please check the values and try again."

    groups = []
    for group_name, prefix in [
        ("Mean measurements", "mean"),
        ("Standard error measurements", "error"),
        ("Worst / largest measurements", "worst"),
    ]:
        fields = []
        for name in feature_names:
            if name.startswith(prefix + "_"):
                fields.append({"name": name, "label": feature_descriptions[name], "value": values[name]})
        groups.append({"title": group_name, "fields": fields})

    return render_template("index.html", groups=groups, result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
