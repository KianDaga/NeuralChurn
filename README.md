# User Churn Early Warning System

A production-style system that predicts churn using the public Telco Customer Churn dataset (OpenML 42178). The system includes a dataset downloader, a TensorFlow/Keras model, a FastAPI inference server, and n8n automations to trigger retention actions.

This repo is sanitized for public GitHub use: all webhook URLs, email addresses, and credential IDs in workflow JSON files are replaced with placeholders.

## Architecture (Text Diagram)

[Data Generator] -> [CSV Dataset] -> [Training Pipeline] -> [Saved Model + Scaler]
                                          |
                                          v
                                 [FastAPI Inference API]
                                          |
                                          v
                                 [n8n Daily Scan Workflow]
                                          |
                                          v
                               [Alerts + Retention Actions]

## How Churn Is Defined
The dataset provides a `Churn` label (Yes/No), which is used as the prediction target.

## Model Design
- Inputs: Telco customer features (demographics, services, contract, billing)
- Model: Dense(64) -> Dense(32) -> Dense(1, sigmoid)
- Metrics: AUC, Precision, Recall
- Class imbalance handled with class weights
- Standardization with mean/std from training split stored in `training/preprocess_meta.json`

## Project Structure
```
churn-ai/
├── data/
│   └── churn_dataset.csv
├── training/
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── scaler.npz
├── api/
│   ├── main.py
│   └── requirements.txt
├── n8n/
│   ├── workflow_daily_scan.json
│   ├── workflow_retention.json
│   └── workflow_weekly_report.json
└── README.md
```

## How To Generate Data
```
python3 training/generate_dataset.py
```
This downloads the Telco Customer Churn dataset from OpenML and saves it as `data/churn_dataset.csv`.

For demo workflows, you can shrink the dataset to a few rows (useful for testing).

## How To Train
```
python3 training/train_model.py
```
Artifacts:
- `training/churn_model.h5`
- `training/preprocess_meta.json`

## How To Evaluate
```
python3 training/evaluate.py
```
Outputs AUC, precision, and recall on a validation split.

## How To Run The API
```
pip3 install -r api/requirements.txt
uvicorn api.main:app --reload
```
Note: TensorFlow does not currently ship wheels for Python 3.13. Use Python 3.12 (or 3.11) for the API/training environment.

## Prediction Storage (SQLite)
Predictions are stored in a local SQLite DB at `data/predictions.db` for weekly reporting.
You can query them via:
```
GET /predictions?range=7d
```

### Example Prediction
Request:
```
POST /predict
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

Response:
```
{
  "user_id": "u123",
  "churn_probability": 0.82,
  "risk_level": "high"
}
```

## n8n Workflows
Import the JSON files from `n8n/` into n8n via **Import Workflow**.

### Credentials & Placeholders
Before running workflows, update these in n8n:
- Discord webhook URL(s)
- Gmail/SMTP credentials for email nodes
- Google Sheets credentials + Sheet ID

All placeholders are intentionally blanked in this repo for safe public sharing.

### Workflow 1 — Daily Churn Scan
- Trigger: Cron daily at 09:00
- Steps: Load user data -> POST /batch_predict -> filter high-risk -> alert + log + tag

### Workflow 2 — Retention Actions
- Trigger: When a user is marked at-risk
- High: Send retention email + create follow-up task
- Medium: Send in-app tip email
- Low: Log only

### Workflow 3 — Weekly Intelligence Report
- Trigger: Weekly Cron
- Steps: Fetch last 7 days -> aggregate counts -> send email/Discord report

## Portfolio Explanation
This project demonstrates end-to-end ML + automation delivery:
- Data engineering and synthetic dataset creation
- Production-ready model training and evaluation
- Real-time inference API with validation
- Business workflows in n8n to automate retention
- Clean repo structure and reproducible pipelines

It shows how to move from behavior data to actionable retention workflows in a SaaS context.

## GitHub Deployment Notes
- Do not commit real webhooks, tokens, or credentials.
- Keep model artifacts (`training/churn_model.h5`, `training/preprocess_meta.json`) and local DB (`data/predictions.db`) out of Git.
- Use the provided `.gitignore`.

## Deployment
### Local (recommended for portfolio demos)
1) Create a Python 3.12 venv and install dependencies:
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r api/requirements.txt
```
2) Download data, train the model, and start the API:
```
python3 training/generate_dataset.py
python3 training/train_model.py
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
3) Run n8n (Docker) with data + persistent storage:
```
docker run -it --rm \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  -v /Users/kian/Documents/userchurndetector/data:/home/node/.n8n-files \
  n8nio/n8n
```
4) Import workflows from `n8n/` and reattach credentials.

### Production notes
- Use a managed DB (Postgres) instead of local SQLite for predictions.
- Run the API behind a reverse proxy (nginx) with TLS.
- Store secrets in environment variables or a secrets manager.
# NeuralChurn
