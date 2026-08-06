from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI(title="Breast Cancer Prediction")

# Load trained model
model = joblib.load("model.pkl")

# Templates & Static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Feature Names
columns = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness',
    'mean concavity',
    'mean concave points',
    'mean symmetry',
    'mean fractal dimension',
    'radius error',
    'texture error',
    'perimeter error',
    'area error',
    'smoothness error',
    'compactness error',
    'concavity error',
    'concave points error',
    'symmetry error',
    'fractal dimension error',
    'worst radius',
    'worst texture',
    'worst perimeter',
    'worst area',
    'worst smoothness',
    'worst compactness',
    'worst concavity',
    'worst concave points',
    'worst symmetry',
    'worst fractal dimension'
]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "columns": columns,
            "prediction": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):

    form = await request.form()

    values = []

    for col in columns:
        values.append(float(form[col]))

    df = pd.DataFrame([values], columns=columns)

    prediction = model.predict(df)[0]

    if prediction == 0:
        result = "🔴 Malignant (Cancer Detected)"
    else:
        result = "🟢 Benign (No Cancer)"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "columns": columns,
            "prediction": result
        }
    )