.PHONY: help build run stop clean dev logs test install check-wsl

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

check-wsl: ## Check if running in WSL
	@if [ -f /proc/version ] && grep -q Microsoft /proc/version; then \
		echo "✅ Running in WSL - Good to go!"; \
	else \
		echo "⚠️  Not running in WSL - some features may not work as expected"; \
	fi

install: ## Install dependencies using poetry
	@echo "Installing dependencies..."
	poetry install

build: ## Build Docker images
	@echo "Building Docker images..."
	@if command -v docker-compose >/dev/null 2>&1; then \
		docker-compose -f docker/docker-compose.yml build --no-cache; \
	else \
		docker compose -f docker/docker-compose.yml build --no-cache; \
	fi

run: ## Run the application in production mode
	@echo "Starting Flight Delay API in production mode..."
	@if command -v docker-compose >/dev/null 2>&1; then \
		docker-compose -f docker/docker-compose.yml up -d; \
	else \
		docker compose -f docker/docker-compose.yml up -d; \
	fi
	@echo "API running at http://localhost:8080"
	@echo "Health check: http://localhost:8080/health"
	@echo "API docs: http://localhost:8080/docs"

dev: ## Run the application in development mode with hot reload
	@echo "Starting Flight Delay API in development mode..."
	@if command -v docker-compose >/dev/null 2>&1; then \
		docker-compose -f docker/docker-compose.dev.yml up --build; \
	else \
		docker compose -f docker/docker-compose.dev.yml up --build; \
	fi
	@echo "Development API running at http://localhost:8080"

run-with-nginx: ## Run with Nginx reverse proxy
	@echo "Starting Flight Delay API with Nginx..."
	docker-compose -f docker/docker-compose.yml --profile production up -d
	@echo "API running at http://localhost (port 80)"

run-with-monitoring: ## Run with monitoring (Prometheus)
	@echo "Starting Flight Delay API with monitoring..."
	docker-compose -f docker/docker-compose.yml --profile monitoring up -d
	@echo "API running at http://localhost:8080"
	@echo "Prometheus running at http://localhost:9090"

stop: ## Stop all containers
	@echo "Stopping all containers..."
	docker-compose -f docker/docker-compose.yml down
	docker-compose -f docker/docker-compose.dev.yml down

clean: ## Clean up containers, images, and volumes
	@echo "Cleaning up Docker resources..."
	docker-compose -f docker/docker-compose.yml down -v --remove-orphans
	docker-compose -f docker/docker-compose.dev.yml down -v --remove-orphans
	docker system prune -f
	docker volume prune -f

logs: ## Show logs from all containers
	docker-compose -f docker/docker-compose.yml logs -f

logs-api: ## Show logs from API container only
	docker-compose -f docker/docker-compose.yml logs -f flight-delay-api

shell: ## Access API container shell
	docker-compose -f docker/docker-compose.yml exec flight-delay-api /bin/bash

test: ## Run tests inside container
	docker-compose -f docker/docker-compose.yml exec flight-delay-api python -m pytest tests/ -v

test-api: ## Test API endpoints
	@echo "Testing API endpoints..."
	@echo "Health check:"
	curl -f http://localhost:8080/health || echo "❌ Health check failed"
	@echo "\nTesting predict endpoint (requires model loaded):"
	curl -X POST http://localhost:8080/model/predict/ \
		-H "Content-Type: application/json" \
		-d '{"dep_delay":5,"sched_dep_time":830,"dep_time":835,"sched_arr_time":1030,"arr_time":1035,"hour":8}' \
		|| echo "❌ Predict endpoint failed (model may not be loaded)"

load-model: ## Load default model
	@echo "Loading default model..."
	curl -X POST http://localhost:8080/model/load/default || echo "❌ Failed to load model"

status: ## Show container status
	@echo "Container status:"
	docker-compose ps
	@echo "\nDocker images:"
	docker images | grep flight-delay

backup-data: ## Backup data and model directories
	@echo "Creating backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz model/ data/
	@echo "Backup created: backup_$(shell date +%Y%m%d_%H%M%S).tar.gz"

setup-wsl: ## Setup WSL Ubuntu-24.04 environment
	@echo "Setting up WSL Ubuntu-24.04 environment..."
	sudo apt-get update
	sudo apt-get install -y curl wget git make
	# Install Docker
	curl -fsSL https://get.docker.com -o get-docker.sh
	sudo sh get-docker.sh
	sudo usermod -aG docker $$USER
	# Install Docker Compose
	sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(shell uname -s)-$(shell uname -m)" -o /usr/local/bin/docker-compose
	sudo chmod +x /usr/local/bin/docker-compose
	# Install Poetry
	curl -sSL https://install.python-poetry.org | python3 -
	@echo "⚠️  Please logout and login again to apply Docker group membership"
	@echo "⚠️  Run 'source ~/.bashrc' or restart terminal"

# WSL specific commands
wsl-setup: setup-wsl ## Alias for setup-wsl

wsl-start-docker: ## Start Docker daemon in WSL
	@echo "Starting Docker daemon in WSL..."
	sudo service docker start
	@echo "Docker daemon started"

wsl-ip: ## Get WSL IP address
	@echo "WSL IP addresses:"
	ip addr show eth0 | grep inet | awk '{print $$2}' | cut -d/ -f1

# Quick start sequence
quick-start: check-wsl build run load-model test-api ## Quick start: build, run, load model and test

# Development workflow
dev-start: check-wsl dev ## Start development environment

# Production deployment
prod-deploy: check-wsl build run-with-nginx ## Deploy in production with Nginx