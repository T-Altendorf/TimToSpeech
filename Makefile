.PHONY: build up down logs restart clean test-local dev dev-down dev-logs dev-restart help

# Detect docker compose command (supports both docker-compose and docker compose)
DOCKER_COMPOSE := $(shell command -v docker-compose 2> /dev/null)
ifndef DOCKER_COMPOSE
	DOCKER_COMPOSE := docker compose
endif

# Compose files
COMPOSE_FILES := -f docker-compose.yml
DEV_COMPOSE_FILES := -f docker-compose.yml -f docker-compose.dev.yml

# Build the Docker image
build:
	$(DOCKER_COMPOSE) $(COMPOSE_FILES) build

# Start the service (production mode - no host port exposure)
up:
	$(DOCKER_COMPOSE) $(COMPOSE_FILES) up -d

# Stop the service
down:
	$(DOCKER_COMPOSE) $(COMPOSE_FILES) down

# View logs
logs:
	$(DOCKER_COMPOSE) $(COMPOSE_FILES) logs -f

# Restart the service
restart: down up

# Clean up everything including volumes
clean:
	$(DOCKER_COMPOSE) $(COMPOSE_FILES) down -v
	docker system prune -f

# Run locally without Docker (for development)
test-local:
	pip install -r requirements.txt
	python app.py

# Development mode commands (with host port exposure)
dev:
	$(DOCKER_COMPOSE) $(DEV_COMPOSE_FILES) up -d

dev-down:
	$(DOCKER_COMPOSE) $(DEV_COMPOSE_FILES) down

dev-logs:
	$(DOCKER_COMPOSE) $(DEV_COMPOSE_FILES) logs -f

dev-restart: dev-down dev

dev-build:
	$(DOCKER_COMPOSE) $(DEV_COMPOSE_FILES) build

# Show help
help:
	@echo "Available commands:"
	@echo ""
	@echo "Production mode (no host port exposure):"
	@echo "  make build       - Build the Docker image"
	@echo "  make up          - Start the service (no host port)"
	@echo "  make down        - Stop the service"
	@echo "  make logs        - View service logs"
	@echo "  make restart     - Restart the service"
	@echo "  make clean       - Stop service and remove volumes"
	@echo ""
	@echo "Development mode (with host port exposure):"
	@echo "  make dev         - Start service with host port exposed"
	@echo "  make dev-down    - Stop dev service"
	@echo "  make dev-logs    - View dev service logs"
	@echo "  make dev-restart - Restart dev service"
	@echo "  make dev-build   - Build dev image"
	@echo ""
	@echo "Other:"
	@echo "  make test-local  - Run locally without Docker"
	@echo "  make help        - Show this help message"
