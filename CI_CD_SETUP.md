# CI/CD Setup Guide - Deploy to Firebase Cloud Run from GitHub

This guide walks you through setting up automated deployment from GitHub to Firebase Cloud Run using GitHub Actions.

## Prerequisites

- GitHub repository with your code
- Google Cloud Project with Firebase enabled
- Service Account with appropriate permissions
- Docker container (Dockerfile already in place)

## Step 1: Create a Google Cloud Service Account

1. **Go to Google Cloud Console:**
   ```
   https://console.cloud.google.com/iam-admin/serviceaccounts
   ```

2. **Create a new service account:**
   - Click "Create Service Account"
   - Name: `github-actions-deployer`
   - Description: "Service account for GitHub Actions CI/CD"
   - Click "Create and Continue"

3. **Grant required roles:**
   - Add the following roles:
     - `Cloud Run Admin` - for deploying to Cloud Run
     - `Artifact Registry Writer` - for pushing Docker images
     - `Service Account User` - for using the service account
   - Click "Continue" then "Done"

## Step 2: Create and Download Service Account Key

1. **Create JSON key:**
   - Click on the newly created service account
   - Go to the "Keys" tab
   - Click "Add Key" → "Create new key"
   - Choose "JSON"
   - Click "Create" and download the file

2. **Save the key securely** - you'll use it in the next step

## Step 3: Add GitHub Secrets

1. **Go to your GitHub repository:**
   ```
   https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions
   ```

2. **Add the following secrets:**

   ### `GCP_PROJECT_ID`
   - Value: Your Google Cloud Project ID (e.g., `my-project-123456`)
   - Get it from: https://console.cloud.google.com/home/dashboard

   ### `GCP_SA_KEY`
   - Value: Paste the entire contents of the JSON key file you downloaded
   - Example:
     ```json
     {
       "type": "service_account",
       "project_id": "...",
       "private_key_id": "...",
       ...
     }
     ```

   ### `GENERATIVE_AI_KEY`
   - Value: Your Google Generative AI API key
   - Get it from: https://aistudio.google.com/app/apikey
   - This will be injected as an environment variable in Cloud Run

3. **Verify all three secrets are added** in the Actions secrets page

## Step 4: Configure Artifact Registry (Optional but Recommended)

1. **Enable Artifact Registry API:**
   ```bash
   gcloud services enable artifactregistry.googleapis.com
   ```

2. **Create an Artifact Registry repository:**
   ```bash
   gcloud artifacts repositories create platground \
     --repository-format=docker \
     --location=us-central1 \
     --description="Platground Docker images"
   ```

## Step 5: Configure Cloud Run Settings

The GitHub Actions workflow automatically handles:
- Building the Docker image
- Pushing to Artifact Registry
- Deploying to Cloud Run with:
  - 512Mi memory
  - 1 CPU
  - Allow unauthenticated access
  - Environment variables (API keys)
  - Auto-scaling up to 100 instances

To customize these settings, edit `.github/workflows/deploy-to-firebase.yml`:

```yaml
--memory 512Mi          # Change memory allocation
--cpu 1                 # Change CPU allocation
--max-instances 100     # Change max scaling
--timeout 3600          # Change timeout (seconds)
```

## Step 6: Trigger Your First Deployment

1. **Push to main branch:**
   ```bash
   git add .
   git commit -m "Setup CI/CD pipeline"
   git push origin main
   ```

2. **Monitor the deployment:**
   - Go to: `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`
   - Click on the workflow run
   - Watch the progress in real-time

3. **Check the deployment summary:**
   - The workflow will output the Cloud Run service URL
   - Click the link to test your deployed API

## Step 7: Test Your Deployed API

Once deployed, test the `/generate_form` endpoint:

```bash
curl -X POST https://your-service-url/generate_form \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "Create a registration form for a tech conference"
  }'
```

Or visit the Swagger UI:
```
https://your-service-url/docs
```

## Monitoring and Logs

**View Cloud Run logs:**
```bash
gcloud run logs read platground --region=us-central1 --limit=50
```

**View deployment history:**
- Go to: https://console.cloud.google.com/run?project=YOUR_PROJECT_ID
- Select the `platground` service
- View revisions, traffic splits, and logs

**Set up alerts:**
- Go to: https://console.cloud.google.com/monitoring
- Create uptime checks and alert policies

## Troubleshooting

### "Deployment failed - authentication error"
- Verify `GCP_SA_KEY` secret is correctly set
- Check that the service account has the required roles

### "Docker image push failed"
- Verify `GCP_PROJECT_ID` is correct
- Ensure Artifact Registry API is enabled
- Check service account has `Artifact Registry Writer` role

### "Service failed to start"
- Check Cloud Run logs: `gcloud run logs read platground --region=us-central1`
- Verify `GENERATIVE_AI_KEY` is set in secrets
- Ensure Dockerfile exposes port 8080

### "Health check failed"
- Add a simple health endpoint (e.g., `/health`)
- Verify the app starts successfully locally first

## Advanced: Custom Domain

To use a custom domain with Cloud Run:

1. **Set up a custom domain:**
   ```bash
   gcloud run domain-mappings create \
     --service=platground \
     --domain=api.yourdomain.com \
     --region=us-central1
   ```

2. **Configure DNS records** according to GCP instructions

3. **Update your workflow** to output the custom domain URL

## Next Steps

1. Customize the workflow for your needs
2. Add more environments (staging, production)
3. Implement manual approval gates for production
4. Set up automated rollback on failure
5. Configure Cloud Monitoring dashboards

For more information:
- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Cloud Run CI/CD Best Practices](https://cloud.google.com/run/docs/continuous-deployment-with-cloud-build)
