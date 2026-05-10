# Makefile for BlossomBot

# Load environment variables from .env file if it exists
-include .env

# --- Variables ---
CLUSTER_NAME    := blossombot-cluster
K8S_DIR         := k8s
NGROK_CONFIG	:= ngrok.yml
NGROK_LOG       := ngrok.log

# DOCKER_HUB_USER is expected to be set in the .env file
PROD_IMAGE_NAME := $(DOCKER_HUB_USER)/blossombot
BUILDER_IMAGE_NAME := $(DOCKER_HUB_USER)/romistral-builder:latest

.PHONY: up create-cluster delete-cluster logs shell ngrok ngrok-down \
        build-app-image load-app-image-to-kind push-app-image \
        build-builder-image load-builder-image-to-kind push-builder-image \
        status update install-ingress build-and-push-docker-images \
        restart-chat restart-ollama restart-qdrant restart-rabbitmq \
        restart-builder restart-all clean-apps pause-cluster resume-cluster

# ==============================================================================
# KIND LOCAL SIMULATION
# ==============================================================================

# --- Pause & Resume Cluster ---

# Oprește toate aplicațiile pentru a elibera resursele (RAM/CPU)
pause-cluster:
	@echo "--- Scaling down all deployments to 0 (Pausing cluster) ---"
	@kubectl scale deployment --all --replicas=0
	@echo "Cluster is now sleeping. Zero resources consumed!"

# Trezește toate aplicațiile la viață
resume-cluster:
	@echo "--- Scaling up all deployments to 1 (Waking up cluster) ---"
	@kubectl scale deployment --all --replicas=1
	@echo "Cluster is booting up. Run 'make status' to check readiness."

# Creates cluster, installs ingress, builds image, loads it, and deploys the app.
up:
	@make build-and-push-docker-images
	@echo "--- Deploying application ---"
	@sleep 30
	@make update
	@echo "Deployment done!"

update:
	@kubectl apply -f $(K8S_DIR)/ --recursive

status:
	@kubectl get all -o wide

create-cluster:
	@echo "--- Creating kind cluster with Ingress ports ---"
	@kind create cluster --name $(CLUSTER_NAME) --config kind-config.yaml
	@make install-ingress

delete-cluster:
	@echo "--- Deleting kind cluster ---"
	@kind delete cluster --name $(CLUSTER_NAME) || true

install-ingress:
	@echo "--- Installing NGINX Ingress Controller ---"
	@kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
	@echo "--- Waiting for Ingress controller deployment to be ready ---"
	@sleep 20
	@kubectl wait --namespace ingress-nginx \
	  --for=condition=available deployment/ingress-nginx-controller \
	  --timeout=120s

build-and-push-docker-images:
	@echo "--- Building Docker Images ---"
	@make build-app-image
	@make build-builder-image
	@echo "--- Loading Images into Kind cluster ---"
	@make load-app-image-to-kind
	@make load-builder-image-to-kind
	@echo "--- Pushing Images to Docker Hub ---"
	@make push-app-image
	@make push-builder-image

restart-chat:
	@echo "--- Restarting Chat App ---"
	@kubectl rollout restart deployment chat-app

restart-ollama:
	@echo "--- Restarting Ollama (Reloading models in RAM) ---"
	@kubectl rollout restart deployment ollama

restart-qdrant:
	@echo "--- Restarting Qdrant Vector DB ---"
	@kubectl rollout restart deployment qdrant

restart-rabbitmq:
	@echo "--- Restarting RabbitMQ ---"
	@kubectl rollout restart deployment rabbitmq

# Resetarea Builder-ului (Job-ul trebuie șters pentru a rula din nou)
restart-builder:
	@echo "--- Resetting Romistral Builder Job ---"
	@kubectl delete job romistral-builder --ignore-not-found

restart-all:
	@echo "--- RESTARTING ALL COMPONENTS ---"
	@make restart-chat
	@make restart-ollama
	@make restart-qdrant
	@make restart-rabbitmq
	@make restart-builder
	@make update
	@echo "All deployments have been triggered to restart."

# ==============================================================================
# DOCKER HUB / PRODUCTION WORKFLOW (for building and pushing, or for local Kind load)
# ==============================================================================

# Builds the application image.
build-app-image:
	@echo "--- Building Docker image for local use ---"
	@docker build -t $(PROD_IMAGE_NAME) .

# Loads the application image into the Kind cluster.
load-app-image-to-kind:
	@echo "--- Loading Docker image into Kind cluster ---"
	@kind load docker-image $(PROD_IMAGE_NAME) --name $(CLUSTER_NAME)

push-app-image:
	@echo "--- Pushing Docker image to Docker Hub ---"
	@docker login
	@docker push $(PROD_IMAGE_NAME)

build-builder-image:
	@echo "--- Building Romistral Builder image ---"
	@docker build -t $(BUILDER_IMAGE_NAME) -f Dockerfile.builder .

load-builder-image-to-kind:
	@kind load docker-image $(BUILDER_IMAGE_NAME) --name $(CLUSTER_NAME)

push-builder-image:
	@echo "--- Pushing Romistral Builder image to Docker Hub ---"
	@docker login
	@docker push $(BUILDER_IMAGE_NAME)

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

