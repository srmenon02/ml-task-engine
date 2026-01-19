#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
celery -A workers.celery_app worker --loglevel=info --pool=solo