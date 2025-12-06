# Cloud Run Deployment Guide

This file contains step-by-step commands to build and deploy the FastAPI app to Google Cloud Run.

## Prerequisites
- Install the Google Cloud SDK and authenticate: `gcloud auth login` and `gcloud config set project YOUR_GCP_PROJECT_ID`.
- Enable Cloud Run and Cloud Build APIs: `gcloud services enable run.googleapis.com cloudbuild.googleapis.com`.
- (Optional) Install Docker Desktop for local container testing.

## Build and deploy (PowerShell)

```powershell
$PROJECT_ID = "tofeset-e2e76"
$REGION = "us-central1"
$IMAGE = "gcr.io/$PROJECT_ID/platground:latest"

# Build and push the container image
gcloud builds submit --project $PROJECT_ID --tag $IMAGE .

# Deploy to Cloud Run (allow unauthenticated access)
gcloud run deploy platground \
  --image $IMAGE \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-secrets GENERATIVE_AI_KEY=GENERATIVE_AI_KEY:latest \
  --port 8000 \
  --project $PROJECT_ID```

##Check gcloud service:
gcloud run services logs read platground --project $PROJECT_ID --region $REGION

## Local testing

- Run with uvicorn (no Docker):

```powershell
python -m uvicorn app:API --host 0.0.0.0 --port 8000
```

- Build and run with Docker:

```powershell
docker build -t platground:local .
docker run -p 8000:8000 platground:local
# then visit http://localhost:8000/generate_form
```

## Notes
- `requirements.txt` contains package names from `main.py` imports but you may need to adjust exact PyPI package names for some langchain-related packages.
- For Google API access you may need to provide credentials via environment variables or mounted service account keys.

## important
- get list of all images on gcloud
gcloud container images list