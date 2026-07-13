MARTA_ROOT := $(shell pwd)/..

.PHONY: dev backend-only frontend-only install

dev:
	docker-compose up --build

backend-only:
	cd backend && MARTA_ROOT=$(MARTA_ROOT) PYTHONPATH=$(MARTA_ROOT) uvicorn main:app --reload --port 8000

frontend-only:
	cd frontend && npm run dev

install:
	cd frontend && npm install
	pip install -r backend/requirements.txt
