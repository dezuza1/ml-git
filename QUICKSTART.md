# Quick Start Guide

## Step 1: Install Dependencies (First Time Only)

If you haven't installed the required packages yet:

```bash
pip install -r requirements.txt
```

Or if using the virtual environment (recommended):

```bash
# Windows
venv\Scripts\activate
pip install -r requirements.txt
```

## Step 2: Train Models (If Not Already Done)

The project comes with pre-trained models. If you need to retrain:

```bash
python backend/train_models.py
```

## Step 3: Start the Backend Server

### Option A: Using the batch file (Windows)
```bash
start_backend.bat
```

### Option B: Manual command
```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Option C: Using the Python file directly
```bash
cd backend
python main.py
```

The backend will start at: **http://127.0.0.1:8000**

## Step 4: Open the Frontend

### Option A: Using the batch file (Windows)
```bash
start_frontend.bat
```

### Option B: Open directly
Simply double-click `frontend/index.html` in your file explorer.

### Option C: Using a local server (recommended for production)
```bash
cd frontend
python -m http.server 3000
```
Then navigate to http://localhost:3000

## Verify Installation

1. Open your browser to the frontend
2. You should see 5 model buttons at the top
3. Click "KNN (Iris)" button
4. Click "Fill examples" to populate fields
5. Click "Predict" to test the model

If you see a prediction result, everything is working! 🎉

## Troubleshooting

### Backend won't start
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that model files exist in `backend/models/` folder
- Try running: `python backend/train_models.py` to regenerate models

### Frontend shows "Failed to load model definitions"
- Make sure the backend is running on port 8000
- Check that CORS is enabled in the backend
- Verify the API_BASE URL in `index.html` matches your backend URL

### Prediction fails
- Ensure all input fields are filled
- Check that values are within the suggested ranges
- Look at the browser console (F12) for error details

## Next Steps

- Explore each of the 5 models
- Try different input values
- Check the API documentation at http://127.0.0.1:8000/docs
- Modify the code to add your own models!

---

**Enjoy exploring ML Model Predictor!** 🚀
