from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from pathlib import Path

app = FastAPI(title="ML Model API - 5 Models", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def _load_model(filename: str):
    """Load a pickled model from file"""
    path = MODELS_DIR / filename
    if not path.exists():
        raise RuntimeError(f"Model file not found: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


# Load all 5 models
knn_model = _load_model("knn_model.pkl")
logistic_model = _load_model("logistic_model.pkl")
svm_scaler_model = _load_model("svm_model.pkl")  # Returns (scaler, model) tuple
linear_model = _load_model("linear_model.pkl")
mlr_model = _load_model("mlr_model.pkl")

# Extract scaler and model for SVM
svm_scaler, svm_model = svm_scaler_model


# ==================== Input Schemas ====================

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class CaliforniaBinaryInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float


class DiabetesInput(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class CaliforniaInput(BaseModel):
    MedInc: float  # median income
    HouseAge: float  # median house age
    AveRooms: float  # average number of rooms
    AveBedrms: float  # average number of bedrooms
    Population: float  # block population
    AveOccup: float  # average house occupancy
    Latitude: float  # latitude
    Longitude: float  # longitude


# ==================== Data Preparation Functions ====================

def prepare_iris(data: IrisInput) -> np.ndarray:
    return np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]], dtype=float)


def prepare_california_binary(data: CaliforniaBinaryInput) -> np.ndarray:
    return np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude,
    ]], dtype=float)


def prepare_wine(data: WineInput) -> np.ndarray:
    return np.array([[
        data.alcohol,
        data.malic_acid,
        data.ash,
        data.alcalinity_of_ash,
        data.magnesium,
        data.total_phenols,
        data.flavanoids,
        data.nonflavanoid_phenols,
        data.proanthocyanins,
        data.color_intensity,
        data.hue,
        data.od280_od315_of_diluted_wines,
        data.proline,
    ]], dtype=float)


def prepare_diabetes(data: DiabetesInput) -> np.ndarray:
    return np.array([[
        data.age,
        data.sex,
        data.bmi,
        data.bp,
        data.s1,
        data.s2,
        data.s3,
        data.s4,
        data.s5,
        data.s6,
    ]], dtype=float)


def prepare_california(data: CaliforniaInput) -> np.ndarray:
    return np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude,
    ]], dtype=float)


# ==================== Model Metadata for Frontend ====================

MODEL_SPECS = {
    "knn": {
        "title": "KNN (Iris)",
        "endpoint": "/predict/knn",
        "fields": [
            {"key": "sepal_length", "label": "Sepal Length (cm)", "min": 4.3, "max": 7.9, "example": 5.1},
            {"key": "sepal_width", "label": "Sepal Width (cm)", "min": 2.0, "max": 4.4, "example": 3.5},
            {"key": "petal_length", "label": "Petal Length (cm)", "min": 1.0, "max": 6.9, "example": 1.4},
            {"key": "petal_width", "label": "Petal Width (cm)", "min": 0.1, "max": 2.5, "example": 0.2},
        ],
        "labels": ["Setosa", "Versicolor", "Virginica"]
    },
    "logistic": {
        "title": "Logistic Regression (California Housing Binary)",
        "endpoint": "/predict/logistic",
        "fields": [
            {"key": "MedInc", "label": "Median Income", "min": 0.5, "max": 15.0, "example": 3.5},
            {"key": "HouseAge", "label": "House Age (years)", "min": 1.0, "max": 52.0, "example": 25.0},
            {"key": "AveRooms", "label": "Avg Rooms", "min": 1.0, "max": 20.0, "example": 5.5},
            {"key": "AveBedrms", "label": "Avg Bedrooms", "min": 0.5, "max": 8.0, "example": 1.1},
            {"key": "Population", "label": "Population", "min": 3.0, "max": 10000.0, "example": 1200.0},
            {"key": "AveOccup", "label": "Avg Occupancy", "min": 0.5, "max": 20.0, "example": 3.0},
            {"key": "Latitude", "label": "Latitude", "min": 32.5, "max": 42.0, "example": 37.5},
            {"key": "Longitude", "label": "Longitude", "min": -124.5, "max": -114.3, "example": -122.0},
        ],
        "labels": ["Lower Price Band", "Higher Price Band"]
    },
    "svm": {
        "title": "SVM (Wine)",
        "endpoint": "/predict/svm",
        "fields": [
            {"key": "alcohol", "label": "alcohol", "min": 11.03, "max": 14.83, "example": 13.0},
            {"key": "malic_acid", "label": "malic acid", "min": 0.74, "max": 5.8, "example": 2.0},
            {"key": "ash", "label": "ash", "min": 1.36, "max": 3.23, "example": 2.3},
            {"key": "alcalinity_of_ash", "label": "alcalinity of ash", "min": 10.6, "max": 30.0, "example": 19.5},
            {"key": "magnesium", "label": "magnesium", "min": 70.0, "max": 162.0, "example": 99.0},
            {"key": "total_phenols", "label": "total phenols", "min": 0.98, "max": 3.88, "example": 2.3},
            {"key": "flavanoids", "label": "flavanoids", "min": 0.34, "max": 5.08, "example": 2.0},
            {"key": "nonflavanoid_phenols", "label": "nonflavanoid phenols", "min": 0.13, "max": 0.66, "example": 0.3},
            {"key": "proanthocyanins", "label": "proanthocyanins", "min": 0.41, "max": 3.58, "example": 1.6},
            {"key": "color_intensity", "label": "color intensity", "min": 1.28, "max": 13.0, "example": 5.0},
            {"key": "hue", "label": "hue", "min": 0.48, "max": 1.71, "example": 1.04},
            {"key": "od280_od315_of_diluted_wines", "label": "od280/od315", "min": 1.27, "max": 4.0, "example": 2.87},
            {"key": "proline", "label": "proline", "min": 278.0, "max": 1680.0, "example": 745.0},
        ],
        "labels": ["Class 0", "Class 1", "Class 2"]
    },
    "linear": {
        "title": "Linear Regression (Diabetes)",
        "endpoint": "/predict/linear",
        "fields": [
            {"key": "age", "label": "Age", "min": -0.107, "max": 0.11, "example": 0.05},
            {"key": "sex", "label": "Sex", "min": -0.045, "max": 0.051, "example": 0.01},
            {"key": "bmi", "label": "BMI", "min": -0.09, "max": 0.17, "example": 0.06},
            {"key": "bp", "label": "Blood Pressure", "min": -0.112, "max": 0.132, "example": 0.02},
            {"key": "s1", "label": "S1 (tc)", "min": -0.127, "max": 0.154, "example": 0.0},
            {"key": "s2", "label": "S2 (ldl)", "min": -0.116, "max": 0.198, "example": 0.0},
            {"key": "s3", "label": "S3 (hdl)", "min": -0.102, "max": 0.181, "example": 0.0},
            {"key": "s4", "label": "S4 (tch)", "min": -0.076, "max": 0.185, "example": 0.0},
            {"key": "s5", "label": "S5 (ltg)", "min": -0.126, "max": 0.134, "example": 0.0},
            {"key": "s6", "label": "S6 (glu)", "min": -0.138, "max": 0.136, "example": 0.0},
        ],
        "is_regression": True,
        "output_label": "Disease Progression"
    },
    "mlr": {
        "title": "Multiple Linear Regression (California Housing)",
        "endpoint": "/predict/mlr",
        "fields": [
            {"key": "MedInc", "label": "Median Income", "min": 0.5, "max": 15.0, "example": 3.5},
            {"key": "HouseAge", "label": "House Age (years)", "min": 1.0, "max": 52.0, "example": 25.0},
            {"key": "AveRooms", "label": "Avg Rooms", "min": 1.0, "max": 20.0, "example": 5.5},
            {"key": "AveBedrms", "label": "Avg Bedrooms", "min": 0.5, "max": 8.0, "example": 1.1},
            {"key": "Population", "label": "Population", "min": 3.0, "max": 10000.0, "example": 1200.0},
            {"key": "AveOccup", "label": "Avg Occupancy", "min": 0.5, "max": 20.0, "example": 3.0},
            {"key": "Latitude", "label": "Latitude", "min": 32.5, "max": 42.0, "example": 37.5},
            {"key": "Longitude", "label": "Longitude", "min": -124.5, "max": -114.3, "example": -122.0},
        ],
        "is_regression": True,
        "output_label": "House Price (100k$)"
    }
}


# ==================== API Endpoints ====================

@app.get("/")
def home():
    return {
        "message": "ML Model API - 5 Models",
        "models": list(MODEL_SPECS.keys()),
        "endpoints": [spec["endpoint"] for spec in MODEL_SPECS.values()]
    }


@app.get("/models")
def list_models():
    """Return model specifications for frontend"""
    return MODEL_SPECS


def _predict(model, model_name: str, x: np.ndarray):
    """Generic prediction function"""
    try:
        pred = model.predict(x)[0]
        return {"model": model_name, "prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed for {model_name}: {e}")


@app.post("/predict/knn")
def predict_knn(data: IrisInput):
    x = prepare_iris(data)
    result = _predict(knn_model, "KNN", x)
    labels = ["Setosa", "Versicolor", "Virginica"]
    result["label"] = labels[int(result["prediction"])]
    return result


@app.post("/predict/logistic")
def predict_logistic(data: CaliforniaBinaryInput):
    x = prepare_california_binary(data)
    result = _predict(logistic_model, "Logistic Regression", x)
    label = "Lower Price Band" if result["prediction"] == 0 else "Higher Price Band"
    result["label"] = label
    return result


@app.post("/predict/svm")
def predict_svm(data: WineInput):
    x = prepare_wine(data)
    x_scaled = svm_scaler.transform(x)
    result = _predict(svm_model, "SVM", x_scaled)
    result["label"] = f"Class {int(result['prediction'])}"
    return result


@app.post("/predict/linear")
def predict_linear(data: DiabetesInput):
    x = prepare_diabetes(data)
    result = _predict(linear_model, "Linear Regression", x)
    result["label"] = f"Disease Progression: {result['prediction']:.1f}"
    return result


@app.post("/predict/mlr")
def predict_mlr(data: CaliforniaInput):
    x = prepare_california(data)
    result = _predict(mlr_model, "Multiple Linear Regression", x)
    result["label"] = f"House Price: ${result['prediction']:.2f} (in 100k)"
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
