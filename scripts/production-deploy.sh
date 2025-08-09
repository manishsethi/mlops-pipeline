# scripts/production-deploy.sh
#!/bin/bash

set -e

echo "Starting production deployment..."

# Configuration
DOCKER_IMAGE="your-registry/ml-model-api:latest"
CONTAINER_NAME="ml-model-production"
NETWORK_NAME="ml-network"
VOLUME_NAME="ml-data"

# Create Docker network if it doesn't exist
docker network ls | grep -q $NETWORK_NAME || docker network create $NETWORK_NAME

# Create volume if it doesn't exist
docker volume ls | grep -q $VOLUME_NAME || docker volume create $VOLUME_NAME

# Pull latest image
echo "Pulling latest Docker image..."
docker pull $DOCKER_IMAGE

# Stop and remove existing container
echo "Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Start new container
echo "Starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    --network $NETWORK_NAME \
    -p 5000:5000 \
    --restart unless-stopped \
    -v $VOLUME_NAME:/app/data \
    -v /var/log/ml-api:/app/logs \
    -e ENVIRONMENT=production \
    --health-cmd="curl -f http://localhost:5000/health || exit 1" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-retries=3 \
    $DOCKER_IMAGE

# Wait for container to be healthy
echo "Waiting for container to be healthy..."
timeout 60s bash -c 'while [[ "$(docker inspect --format="{{.State.Health.Status}}" '$CONTAINER_NAME')" != "healthy" ]]; do sleep 2; done'

echo "Deployment completed successfully!"
echo "API is available at http://localhost:5000"
