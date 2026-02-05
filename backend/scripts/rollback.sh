#!/bin/bash
set -e

ENVIRONMENT=${1:-staging}
PREVIOUS_VERSION=${2}

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "Usage: ./rollback.sh <environment> <previous_version>"
    exit 1
fi

echo "Rolling back $ENVIRONMENT to version $PREVIOUS_VERSION"

export VERSION=$PREVIOUS_VERSION

read -p "Rollback database migrations? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose.prod.yml run --rm api \
        alembic downgrade -1
fi

docker-compose -f docker-compose.prod.yml up -d

echo "Rollback completed!"