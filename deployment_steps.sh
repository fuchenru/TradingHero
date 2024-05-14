gcloud config set project adsp-capstone-trading-hero

cd TradingHero


## Clean-up (to get fresh experience) and remove all unused and dangling images
docker system prune -a -f


# Enable Artifact Registry
gcloud services enable artifactregistry.googleapis.com


# Verify / list repositories
gcloud artifacts repositories list


# Create a repository (if it does  not exist)
gcloud artifacts repositories create tradinghero --repository-format=docker --location=us --description="trading hero test push"


# Build Docker image
docker build -t us-docker.pkg.dev/adsp-capstone-trading-hero/tradinghero/app:latest .


# Push Docker image into artifact registry
docker push us-docker.pkg.dev/adsp-capstone-trading-hero/tradinghero/app:latest


# Enable  Cloud Run API
gcloud services enable run.googleapis.com


# Go to Artifact Registry select repository -> app -> the latest image and deploy on cloud run

# Delete Streamlit app when no longer needed
gcloud run services delete APP-NAME --region=us-central1 --quiet
