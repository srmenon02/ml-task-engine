#!/bin/bash
set -e

echo "Running database migrations..."

cd /app/backend

echo "Waiting for database..."
while ! pg_isready -h ${DB_HOST:-postgres} -p ${DB_PORT:-5432} -U ${POSTGRES_USER}; do
    sleep 1
done

echo "Database is ready. Running Alembic migrations..."
alembic upgrade head

echo "Migrations complete!"