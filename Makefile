.PHONY: infra api worker test

# Start Postgres + Redis (free, local)
infra:
	docker compose -f docker/docker-compose.yml up -d

# Stop infrastructure
infra-down:
	docker compose -f docker/docker-compose.yml down

# Run FastAPI server
api:
	cd backend && uvicorn app.main:app --reload --port 8000

# Run Celery worker
worker:
	cd backend && celery -A app.workers.training_worker.celery_app worker --loglevel=info

# Install dependencies
install:
	cd backend && pip install -r requirements.txt

# Run tests
test:
	cd backend && pytest tests/ -v
