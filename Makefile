.PHONY: build up down logs restart clean test-local help

# Detect docker compose command (supports both docker-compose and docker compose)
DOCKER_COMPOSE := $(shell command -v docker-compose 2> /dev/null)
ifndef DOCKER_COMPOSE
	DOCKER_COMPOSE := docker compose
endif

# Build the Docker image
build:
	$(DOCKER_COMPOSE) build

# Start the service
up:
	$(DOCKER_COMPOSE) up -d

# Stop the service
down:
	$(DOCKER_COMPOSE) down

# View logs
logs:
	$(DOCKER_COMPOSE) logs -f

# Restart the service
restart: down up

# Clean up everything including volumes
clean:
	$(DOCKER_COMPOSE) down -v
	docker system prune -f

# Run locally without Docker (for development)
test-local:
	pip install -r requirements.txt
	python app.py

# Show help
help:
	@echo "Available commands:"
	@echo "  make build       - Build the Docker image"
	@echo "  make up          - Start the service in detached mode"
	@echo "  make down        - Stop the service"
	@echo "  make logs        - View service logs"
	@echo "  make restart     - Restart the service"
	@echo "  make clean       - Stop service and remove volumes"
	@echo "  make test-local  - Run locally without Docker"
	@echo "  make help        - Show this help message"
