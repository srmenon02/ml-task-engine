#!/bin/bash
set -e

echo "Running database initialization..."
cd /app

python -c "
import sys
sys.path.insert(0, '/app')
from models import init_db
init_db()
print('Database initialized successfully')
"

echo "Starting API server..."
exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}