.PHONY: build up down logs restart clean test-local help

# Build the Docker image
build:
	docker-compose build

# Start the service
up:
	docker-compose up -d

# Stop the service
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Restart the service
restart: down up

# Clean up everything including volumes
clean:
	docker-compose down -v
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
