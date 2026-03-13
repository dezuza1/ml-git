# ML Model Predictor - 5 Models

A comprehensive machine learning web application featuring 5 different models with an interactive frontend. This project demonstrates various ML algorithms applied to different datasets with a beautiful, user-friendly interface.

## 🌟 Features

- **5 Pre-trained ML Models**: KNN, Logistic Regression, SVM, and 2 Linear Regression models
- **Multiple Datasets**: Iris, Wine (binary & 3-class), Diabetes, and California Housing
- **Interactive Frontend**: Clean, modern UI with separate buttons for each model
- **Real-time Validation**: Input validation with min/max range checking
- **RESTful API**: FastAPI backend with CORS support
- **Example Data**: Quick-fill buttons to test models with sample data

## 📁 Project Structure

```
ml_project/
├── backend/
│   ├── main.py              # FastAPI application with 5 endpoints
│   ├── train_models.py      # Script to train and save models
│   └── models/              # Trained model files (.pkl)
│       ├── knn_model.pkl
│       ├── logistic_model.pkl
│       ├── svm_model.pkl
│       ├── linear_model.pkl
│       └── mlr_model.pkl
├── frontend/
│   └── index.html           # Single-page web application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Models Overview

| Model | Algorithm | Dataset | Type | Features |
|-------|-----------|---------|------|----------|
| **KNN** | K-Nearest Neighbors | Iris | Classification (3-class) | 4 features (sepal/petal measurements) |
| **Logistic** | Logistic Regression | Wine Binary | Classification (2-class) | 13 features (Class 0 detection) |
| **SVM** | Support Vector Machine | Wine | Classification (3-class) | 13 features (wine properties) |
| **Linear** | Linear Regression | Diabetes | Regression | 10 features (patient data) |
| **MLR** | Multiple Linear Regression | California Housing | Regression | 8 features (housing data) |

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (if not already trained)
   ```bash
   python backend/train_models.py
   ```
   This will create 5 .pkl files in the `backend/models/` directory.

## 🎯 Usage

### 1. Start the Backend Server

From the project root directory:

```bash
cd backend
python main.py
```

Or use uvicorn directly:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://127.0.0.1:8000`

### 2. Open the Frontend

Simply open `frontend/index.html` in your web browser, or serve it with a local server:

```bash
# Using Python's built-in server
cd frontend
python -m http.server 3000
```

Then navigate to `http://localhost:3000`

### 3. Use the Application

1. **Select a Model**: Click one of the 5 model buttons at the top
2. **Fill Input Fields**: Enter values for all required features
3. **Validate**: Check that values are within the suggested ranges
4. **Predict**: Click the "Predict" button to get results
5. **Quick Test**: Use "Fill examples" to populate fields with sample data

## 🔌 API Endpoints

### Get All Models
```
GET /models
```
Returns specifications for all 5 models including field definitions and ranges.

### Predictions

```
POST /predict/knn
POST /predict/logistic
POST /predict/svm
POST /predict/linear
POST /predict/mlr
```

Each endpoint accepts JSON with the appropriate features and returns a prediction.

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict/knn \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

### Example Response

```json
{
  "model": "KNN",
  "prediction": 0,
  "label": "Setosa"
}
```

## 🎨 Frontend Features

- **Responsive Design**: Works on desktop and mobile devices
- **Beautiful Gradients**: Modern glassmorphism design
- **Live Validation**: Real-time input validation with error messages
- **Range Indicators**: Shows expected min/max values for each field
- **Status Feedback**: Clear success/error messages for predictions

## 🧪 Model Performance

Models are trained on sklearn datasets with default train/test splits:

- **KNN (Iris 3-class)**: ~100% accuracy
- **Logistic Regression (Wine Binary - Class 0 Detection)**: ~100% accuracy
- **SVM (Wine 3-class)**: ~100% accuracy (with StandardScaler)
- **Linear Regression (Diabetes)**: R² ~0.45
- **Multiple Linear Regression (California Housing)**: R² ~0.58

## 🛠️ Technologies Used

- **Backend**: FastAPI, Python, scikit-learn, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Model Serialization**: pickle
- **Data Processing**: NumPy arrays

## 📝 Notes

- The SVM model uses StandardScaler for feature scaling (included in the pickle)
- Input validation ranges are based on training data statistics
- All models use sklearn's built-in datasets
- CORS is enabled for all origins (suitable for development)

## 🔧 Customization

### Adding a New Model

1. Train your model and save it to `backend/models/your_model.pkl`
2. Add input schema (Pydantic BaseModel) in `main.py`
3. Add model spec to `MODEL_SPECS` dictionary
4. Create a prediction endpoint
5. Update frontend to add a new button

### Changing the Port

Backend: Edit `uvicorn.run()` in `main.py` or use CLI flags

```bash
uvicorn backend.main:app --port 8080
```

Frontend: Update `API_BASE` constant in `index.html`

## 📄 License

This project is open source and available for educational purposes.

## 🤝 Contributing

Feel free to fork, modify, and use this project as a template for your own ML applications!

## 📧 Support

For issues or questions, please open an issue in the repository.

---

**Happy Predicting! 🎉**
