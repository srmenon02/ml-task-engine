#!/bin/bash
set -e

ENVIRONMENT=${1:-staging}
BASE_URL="https://${ENVIRONMENT}.your-domain.com"

echo "Running smoke tests against $BASE_URL"

echo "Testing /health endpoint..."
curl -f $BASE_URL/health || exit 1

echo "Testing /api/v1 endpoint..."
curl -f -H "Authorization: Bearer $SMOKE_TEST_API_KEY" \
     $BASE_URL/api/v1/jobs?page=1&page_size=10 || exit 1

echo "Testing /metrics endpoint..."
curl -f $BASE_URL/metrics || exit 1

echo "All smoke tests passed!"