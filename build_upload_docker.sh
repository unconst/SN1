#!/bin/bash
set -e

# Configuration
DOCKER_USERNAME="thebes1618"
IMAGE_NAME="sn1"
DOCKERFILE="Dockerfile"

# Get the latest git commit hash
GIT_COMMIT=$(git rev-parse --short HEAD)

# Build the Docker image
echo "Building Docker image: $DOCKER_USERNAME/$IMAGE_NAME:latest"
docker build -f $DOCKERFILE -t $DOCKER_USERNAME/$IMAGE_NAME:latest .
docker build -f $DOCKERFILE -t $DOCKER_USERNAME/$IMAGE_NAME:$GIT_COMMIT .

# Push the Docker image to Docker Hub
echo "Pushing Docker image: $DOCKER_USERNAME/$IMAGE_NAME:latest"
docker push $DOCKER_USERNAME/$IMAGE_NAME:latest
docker push $DOCKER_USERNAME/$IMAGE_NAME:$GIT_COMMIT

echo "Script finished successfully." 