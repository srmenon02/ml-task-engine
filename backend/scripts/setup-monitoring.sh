#!/bin/bash
set -e

echo "Setting up monitoring stack..."

docker run -d \
    --name prometheus \
    --network taskengine-network \
    -p 9090:9090 \
    -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus

docker run -d \
    --name grafana \
    --network taskengine-network \
    -p 3000:3000 \
    -v grafana-storage:/var/lib/grafana \
    grafana/grafana

echo "Monitoring stack deployed!"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/admin)"