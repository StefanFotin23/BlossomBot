# Makefile for BlossomBot

# Load environment variables from .env file if it exists
-include .env

# --- Variables ---
CLUSTER_NAME    := blossombot-cluster
K8S_DIR         := k8s
NGROK_CONFIG=ngrok.yml
NGROK_LOG       := ngrok.log

# DOCKER_HUB_USER is expected to be set in the .env file
PROD_IMAGE_NAME := $(DOCKER_HUB_USER)/blossombot

.PHONY: up down logs shell ngrok ngrok-down build-image status

# ==============================================================================
# KIND LOCAL SIMULATION
# ==============================================================================

# Creates cluster, installs ingress, builds image, loads it, and deploys the app.
up:
	@make down
	@echo "--- Creating kind cluster with Ingress ports ---"
	@kind create cluster --name $(CLUSTER_NAME) --config kind-config.yaml
	@echo "--- Installing NGINX Ingress Controller ---"
	@kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
	@echo "--- Waiting for Ingress controller deployment to be ready ---"
	@sleep 20
	@kubectl wait --namespace ingress-nginx \
	  --for=condition=available deployment/ingress-nginx-controller \
	  --timeout=120s
	@make build-image
	@echo "--- Deploying application ---"
	@sleep 30
	@make update
	@echo "Deployment done!"

update:
	@kubectl apply -f $(K8S_DIR)/

status:
	@kubectl get all --all-namespaces -o wide

# Deletes the kind cluster and stops the ngrok tunnel.
down:
	@echo "--- Deleting kind cluster ---"
	@kind delete cluster --name $(CLUSTER_NAME) || true

# ==============================================================================
# DOCKER HUB / PRODUCTION WORKFLOW
# ==============================================================================

# Builds and pushes the image to Docker Hub.
build-image:
	@echo "--- Building Docker image for local use ---"
	@docker build -t $(PROD_IMAGE_NAME) .
	@docker login
	@docker push $(PROD_IMAGE_NAME)

# ==============================================================================
# UTILITIES
# ==============================================================================

# Tails the logs of the chat-app pod.
logs:
	@kubectl logs -f $$(kubectl get pods -l app=chat-app -o jsonpath='{.items[0].metadata.name}')

# Opens a shell inside the chat-app pod.
shell:
	@kubectl exec -it $$(kubectl get pods -l app=chat-app -o jsonpath='{.items[0].metadata.name}') -- /bin/sh

# Starts ngrok and captures the public URL.
ngrok: ngrok-down
	@echo "--- Starting ngrok using $(NGROK_CONFIG) ---"
	@ngrok start --all --config $(NGROK_CONFIG) > $(NGROK_LOG) 2>&1 &
	@sleep 3
	@cat ngrok.log

# Stop any running ngrok process
ngrok-down:
	@bash scripts/ngrok_down.sh
	@rm -f $(NGROK_LOG)

