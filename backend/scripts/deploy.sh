#!/bin/bash
set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "Deploying ML Task Engine to $ENVIRONMENT (version: $VERSION)"

source .env.$ENVIRONMENT

docker-compose -f docker-compose.prod.yml pull

docker-compose -f docker-compose.prod.yml run --rm api alembic upgrade head

docker-compose -f docker-compose.prod.yml up -d --no-deps --build api
docker-compose -f docker-compose.prod.yml up -d --no-deps --build worker

echo "Waiting for services to be healthy..."
sleep 10

./scripts/smoke-tests.sh $ENVIRONMENT

echo "Deployment completed successfully!"