# Quick Start: Deploy to Firebase Cloud Run

## ⚡ Quick Setup (5 minutes)

### 1. Get your Google Cloud Project ID
```bash
gcloud config list --format 'value(core.project)'
```

### 2. Create a Service Account
```bash
# Create service account
gcloud iam service-accounts create github-actions-deployer \
  --display-name="GitHub Actions Deployer"

# Grant roles
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member=serviceAccount:github-actions-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/run.admin

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member=serviceAccount:github-actions-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/artifactregistry.writer

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member=serviceAccount:github-actions-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/iam.serviceAccountUser
```

### 3. Create and Download the Key
```bash
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 4. Add GitHub Secrets
Go to: `https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions`

Add three secrets:
- `GCP_PROJECT_ID` = Your project ID (e.g., `my-project-123456`)
- `GCP_SA_KEY` = Contents of `key.json` (the entire JSON file)
- `GENERATIVE_AI_KEY` = Your Google Generative AI key from https://aistudio.google.com/app/apikey

### 5. Create Artifact Registry Repository
```bash
gcloud artifacts repositories create platground \
  --repository-format=docker \
  --location=us-central1
```

### 6. Push and Deploy
```bash
git push origin main
```

The workflow will automatically build, push, and deploy!

## 📊 Monitoring

**View real-time logs:**
```bash
gcloud run logs read platground --region=us-central1 --tail
```

**View deployment history:**
```bash
gcloud run revisions list --service=platground --region=us-central1
```

**Visit your API:**
```
https://console.cloud.google.com/run?project=YOUR_PROJECT_ID
```

## 🔧 Troubleshooting

| Error | Solution |
|-------|----------|
| "Error: Credentials not found" | Check `GCP_SA_KEY` secret is properly set |
| "Permission denied" | Ensure service account has all required roles |
| "Failed to push image" | Verify Artifact Registry repository exists |
| "Health check failed" | Check app starts: `python main.py` |

## 📝 Environment Variables on Cloud Run

The workflow automatically sets:
- `GENERATIVE_AI_KEY` - Your API key (from secrets)
- `PORT` - Always 8000 (Cloud Run requirement)

Add more environment variables in the workflow under `--set-env-vars`.

## 🚀 Manual Deployment (Alternative)

If you prefer to deploy manually without CI/CD:

```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build locally
docker build -t platground:latest .

# Tag for Artifact Registry
docker tag platground:latest \
  us-central1-docker.pkg.dev/YOUR_PROJECT_ID/platground/platground-api:latest

# Push
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/platground/platground-api:latest

# Deploy
gcloud run deploy platground \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/platground/platground-api:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GENERATIVE_AI_KEY=YOUR_API_KEY \
  --memory 512Mi \
  --cpu 1
```

For more details, see `CI_CD_SETUP.md`.
