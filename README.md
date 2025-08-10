# mlops-pipeline

This repository contains an end-to-end MLOps pipeline for serving machine learning models with FastAPI, integrated monitoring with Prometheus and Grafana, and automated retraining via GitHub Actions.

---

## Prerequisites

Before running the application locally or contributing, make sure you have:

- **Python 3.12+** installed
- **Docker and Docker Compose** installed for containerized deployment
- (Optional) **Git Large File Storage (Git LFS)** installed if you work with the repo directly and models are stored via LFS
- Basic familiarity with running REST APIs and curl or HTTP clients

---

## Running the Application Locally

### Using Docker Compose (Recommended)

This is the easiest and reproducible way to run the FastAPI service, Prometheus, and Grafana for metrics monitoring.

1. Clone the repository:

```

git clone https://github.com/manishsethi/mlops-pipeline.git
cd mlops-pipeline

```

2. Build and start services:

```

docker-compose up --build

```

3. Once running, services will be available at:

- FastAPI API: [http://localhost:5000](http://localhost:5000)
- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000) (default admin/password: `admin`)

### Running FastAPI App Manually

1. Create and activate a Python virtual environment:

```

python3 -m venv venv
source venv/bin/activate

```

2. Install dependencies:

```

pip install -r requirements.txt

```

3. Run the FastAPI API using Uvicorn:

```

uvicorn src.Fast_api:app --host 0.0.0.0 --port 5000 --reload

```

API will be available at [http://localhost:5000](http://localhost:5000).

---

## Making API Calls

The FastAPI service serves ML models for multiple tasks (Iris classification and Housing price regression). You can test using `curl` or any REST client.

### 1. Health Check

Check if the API is running:

```

curl http://localhost:5000/health

```

Example response:

```

{
"status": "healthy",
"loaded_models": ["iris", "housing"],
"timestamp": "2025-08-10T13:00:00"
}

```

### 2. Metrics Endpoint (for Prometheus)

To view Prometheus metrics in plain text:

```

curl http://localhost:5000/metrics

```

### 3. Prediction Endpoint

Make a POST request to `/predict` with the task and features in JSON.

#### Example: Iris classification prediction

```

curl -X POST "http://localhost:5000/predict" \
-H "Content-Type: application/json" \
-d '{
"task": "iris",
"features": [5.1, 3.5, 1.4, 0.2]
}'

```

Example JSON response:

```

{
"task": "iris",
"prediction": ,
"prediction_probabilities": [[0.98, 0.01, 0.01]],
"response_time_seconds": 0.005,
"timestamp": "2025-08-10T13:00:05"
}

```

#### Example: Housing price regression prediction

```

curl -X POST "http://localhost:5000/predict" \
-H "Content-Type: application/json" \
-d '{
"task": "housing",
"features": [8.3252, 41, 6.98, 1.02, 322, 2.56, 37.88, -122.23]
}'

```

---

## GitHub Actions: Automated Model Retraining Workflow

Automated retraining is set up using GitHub Actions in `.github/workflows/model-retraining.yml`. Key points:

- **Scheduled trigger:** Runs daily at 2 AM UTC to check for drift.
- **Drift detection:** Runs Python code analyzing recent predictions logged to trigger retraining if needed.
- **Retraining:** Retrains Iris and Housing models if drift detected.
- **Model artifact handling:** Uses Git LFS to track large model `.pkl` files.
- **PR creation:** Automatically opens a pull request with updated models and metrics.
- **Manual trigger:** Workflow can be manually started via GitHub Actions UI (`workflow_dispatch`).

To manually run the retraining workflow:

1. Go to the **Actions** tab in this GitHub repo.
2. Select **Automated Model Retraining** workflow.
3. Click **Run workflow** and confirm.

---

## Additional Notes

- Model monitoring metrics are exposed via `/metrics` and can be visualized with Prometheus and Grafana.
- The retraining system is modular and can be extended for additional task types or advanced drift detection.
- Model files are managed with Git LFS to handle large binary files properly in GitHub.

---

If you have any questions or want to contribute, feel free to open issues or pull requests.

---

*This project is maintained by Manish Sethi and the community.*

```