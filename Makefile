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

.PHONY: up create-cluster delete-cluster logs shell ngrok ngrok-down \
        build-app-image load-app-image-to-kind push-app-image \
        status update install-ingress build-and-push-docker-images \
        restart-chat restart-qdrant restart-rabbitmq restart-all \
		clean-apps pause-cluster resume-cluster restart-cluster

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
delete-resources:
	@echo "--- Deleting all Kubernetes resources ---"
	@kubectl delete all --all || true
	@echo "All resources deleted."

up:
	@make build-and-push-docker-images
	@echo "--- Deploying application ---"
	@make update
	@echo "--- Waiting for pods to be ready ---"
	@kubectl wait --for=condition=ready pod -l app=chat-app --timeout=180s || echo "Wait failed, checking status..."
	@make status
	@echo "Deployment done!"

# Indexează datele în Qdrant folosind Job-ul K8s sau direct exec
index-data:
	@echo "--- Starting RAG Indexing Job ---"
	@kubectl apply -f k8s/chat-app/job-index-rag.yaml
	@echo "Job triggered. Check status with: kubectl get jobs"

update:
	@kubectl apply -f $(K8S_DIR)/ --recursive

status:
	@kubectl get all -o wide

create-cluster:
	@echo "--- Creating kind cluster with Ingress ports ---"
	@kind create cluster --name $(CLUSTER_NAME) --config kind-config.yaml
	@make install-ingress
	@kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml

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
	@echo "--- Loading Images into Kind cluster ---"
	@make load-app-image-to-kind
	@echo "--- Pushing Images to Docker Hub ---"
	@make push-app-image

restart-chat:
	@echo "--- Restarting Chat App ---"
	@kubectl rollout restart deployment chat-app

restart-qdrant:
	@echo "--- Restarting Qdrant Vector DB ---"
	@kubectl rollout restart deployment qdrant

restart-rabbitmq:
	@echo "--- Restarting RabbitMQ ---"
	@kubectl rollout restart deployment rabbitmq

restart-all:
	@echo "--- RESTARTING ALL COMPONENTS ---"
	@make restart-chat
	@make restart-qdrant
	@make restart-rabbitmq
	@make update
	@echo "All deployments have been triggered to restart."

restart-cluster: delete-cluster create-cluster up

# ==============================================================================
# DOCKER HUB / PRODUCTION WORKFLOW (for building and pushing, or for local Kind load)
# ==============================================================================

# Builds the application image.
build-app-image:
	@echo "--- Building Docker image using BuildKit for caching ---"
	@DOCKER_BUILDKIT=1 docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t $(PROD_IMAGE_NAME) .

# Loads the application image into the Kind cluster.
load-app-image-to-kind:
	@echo "--- Loading Docker image into Kind cluster ---"
	@kind load docker-image $(PROD_IMAGE_NAME) --name $(CLUSTER_NAME)

push-app-image:
	@echo "--- Pushing Docker image to Docker Hub ---"
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
	@ngrok start --all --config $(NGROK_CONFIG) --log=stdout > $(NGROK_LOG) 2>&1 &
	@sleep 5
	@grep -o "url=https://[a-z0-9.-]*" $(NGROK_LOG) || cat $(NGROK_LOG)

# Stop any running ngrok process
ngrok-down:
	@bash scripts/ngrok_down.sh
	@rm -f $(NGROK_LOG)

