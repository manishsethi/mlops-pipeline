# scripts/deploy.sh
#!/bin/bash

set -e

DOCKER_IMAGE="your-registry/ml-model-api:latest"
CONTAINER_NAME="ml-model-production"

echo "Deploying ML Model API..."

# Pull latest image
docker pull $DOCKER_IMAGE

# Stop existing container if running
docker stop $CONTAINER_NAME || true
docker rm $CONTAINER_NAME || true

# Run new container
docker run -d \
  --name $CONTAINER_NAME \
  -p 5000:5000 \
  --restart unless-stopped \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  $DOCKER_IMAGE

echo "Deployment completed successfully!"

# Health check
sleep 10
if curl -f http://localhost:5000/health; then
  echo "Health check passed!"
else
  echo "Health check failed!"
  exit 1
fi
