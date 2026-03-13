"""
Script to train and save 5 different ML models to .pkl files
"""
import pickle
from pathlib import Path
from sklearn.datasets import load_iris, load_wine, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np

# Directory setup
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

def save_model(model, filename):
    """Save a trained model to a .pkl file"""
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Saved: {filename}")


def train_iris_knn():
    """Train KNN classifier on Iris dataset"""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"KNN (Iris) accuracy: {accuracy:.3f}")
    
    save_model(model, "knn_model.pkl")


def train_california_binary_logistic():
    """Train Logistic Regression on California Housing for binary classification (high vs low price)."""
    data = fetch_california_housing()
    median_price = float(np.median(data.target))
    y_binary = (data.target >= median_price).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, y_binary, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Logistic Regression (California Housing Binary) accuracy: {accuracy:.3f}")

    save_model(model, "logistic_model.pkl")


def train_wine_svm():
    """Train SVM classifier on Wine dataset"""
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Scale features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    print(f"SVM (Wine) accuracy: {accuracy:.3f}")
    
    # Save both scaler and model
    save_model((scaler, model), "svm_model.pkl")


def train_diabetes_linear():
    """Train Linear Regression on Diabetes dataset"""
    data = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2_score = model.score(X_test, y_test)
    print(f"Linear Regression (Diabetes) R² score: {r2_score:.3f}")
    
    save_model(model, "linear_model.pkl")


def train_california_mlr():
    """Train Multiple Linear Regression on California Housing dataset"""
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2_score = model.score(X_test, y_test)
    print(f"Multiple Linear Regression (California Housing) R² score: {r2_score:.3f}")
    
    save_model(model, "mlr_model.pkl")


if __name__ == "__main__":
    print("Training 5 ML models...\n")
    
    train_iris_knn()
    train_california_binary_logistic()
    train_wine_svm()
    train_diabetes_linear()
    train_california_mlr()
    
    print("\n✓ All models trained and saved successfully!")
    print(f"Models saved in: {MODELS_DIR}")
