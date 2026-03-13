"""Validate saved .pkl model artifacts by loading and running sample predictions."""

from __future__ import annotations

import pickle
from pathlib import Path

from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    load_iris,
    load_wine,
)


MODELS_DIR = Path(__file__).parent / "models"


def load_artifact(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_knn() -> None:
    model = load_artifact("knn_model.pkl")
    x = load_iris().data[:1]
    pred = int(model.predict(x)[0])
    print(f"PASS knn_model.pkl -> class {pred}")


def validate_logistic() -> None:
    model = load_artifact("logistic_model.pkl")
    x = fetch_california_housing().data[:1]
    pred = int(model.predict(x)[0])
    print(f"PASS logistic_model.pkl -> class {pred}")


def validate_svm() -> None:
    artifact = load_artifact("svm_model.pkl")
    if not isinstance(artifact, tuple) or len(artifact) != 2:
        raise TypeError("svm_model.pkl must contain a (scaler, model) tuple")
    scaler, model = artifact
    x = load_wine().data[:1]
    pred = int(model.predict(scaler.transform(x))[0])
    print(f"PASS svm_model.pkl -> class {pred}")


def validate_linear() -> None:
    model = load_artifact("linear_model.pkl")
    x = load_diabetes().data[:1]
    pred = float(model.predict(x)[0])
    print(f"PASS linear_model.pkl -> prediction {pred:.3f}")


def validate_mlr() -> None:
    model = load_artifact("mlr_model.pkl")
    x = fetch_california_housing().data[:1]
    pred = float(model.predict(x)[0])
    print(f"PASS mlr_model.pkl -> prediction {pred:.3f}")


def main() -> None:
    checks = [
        validate_knn,
        validate_logistic,
        validate_svm,
        validate_linear,
        validate_mlr,
    ]
    failures = []

    print(f"Validating model artifacts in: {MODELS_DIR}")
    for check in checks:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - print exact failure per model check
            failures.append(f"{check.__name__}: {exc}")

    if failures:
        print("\nValidation failed:")
        for item in failures:
            print(f"- {item}")
        raise SystemExit(1)

    print("\nAll model artifacts validated successfully.")


if __name__ == "__main__":
    main()