# BlossomBot - A Slack-based Flower Concierge

BlossomBot is a Slack-based chatbot that acts as a virtual concierge for a flower shop. It guides users through the process of selecting and purchasing flowers by understanding their needs, providing recommendations, and gathering feedback.

## Architecture

The application is designed to run on Kubernetes and consists of two main services:

-   **`chat-app`**: A Python application built with FastAPI and the Slack Bolt framework. It contains all the business logic for the chatbot, including conversation management, state tracking, and interaction with the Slack API.
-   **`redis`**: A Redis instance used for in-memory data storage. It caches user session data, conversation history, and the user's current state in the conversation flow.

The `chat-app` is exposed to the internet via a Kubernetes service to receive events from Slack's API.

## Conversation Flow

The bot guides the user through a series of states to complete a purchase:

1.  **`GREETING`**: The user starts a conversation with the bot. The bot introduces itself and asks the user to describe their needs.
2.  **`GATHERING_DESCRIPTION`**: The bot prompts the user for details about what they are looking for. It performs a simple validation to check if the description is detailed enough.
3.  **`CONFIRMING_DESCRIPTION`**: Once a satisfactory description is received, the bot asks the user for confirmation.
4.  **`RECOMMENDING`**: After confirmation, the bot presents the user with a list of mock product recommendations.
5.  **`PURCHASE_CONFIRMATION`**: If the user reacts to a recommendation (e.g., with a :+1: emoji), the bot asks for final confirmation to purchase the item.
6.  **`CLOSED`**: If the user confirms the purchase, the bot closes the transaction and the conversation is complete.

The user can type `restart` at any point to reset the conversation and start over.

## Simulating Deployment with kind and Makefile

This project is primarily designed for local simulation using `kind` (Kubernetes in Docker).

**Prerequisites**:
-   [Docker](https://docs.docker.com/get-docker/)
-   [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)
-   [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/)
-   [make](https://www.gnu.org/software/make/)

### Quick Start

1.  **Create your secret file**: Rename `k8s/secrets.yaml.template` to `k8s/secrets.yaml` and fill in your Slack credentials.
2.  **Start the cluster and deploy**:
    ```bash
    make up
    ```
    Your application's Ingress will be accessible on `http://localhost:40080`.

3.  **Connect to Slack using ngrok**:
    ```bash
    make ngrok
    ```
    This will start `ngrok` and display the public HTTPS URL to paste into Slack's Event Subscriptions.

### Makefile Targets

-   `make up`: Creates the `kind` cluster and deploys the app for local simulation.
-   `make down`: Tears down the `kind` cluster and stops `ngrok`.
-   `make push`: Builds and pushes the Docker image to Docker Hub (for production deployment).
-   `make ngrok`: Starts `ngrok` to expose your local cluster.
-   `make logs`: Tails the application logs.
-   `make shell`: Opens a shell in the application pod.

## Production Deployment (with Docker Hub)

While the `Makefile` is optimized for local `kind` simulation, the `make push` command is included to support a workflow for deploying to a real Kubernetes cluster (e.g., GKE, EKS, AKS).

1.  **Set Your Docker Hub Username**: Open the `Makefile` and change the `DOCKER_HUB_USER` variable from `your-docker-hub-username` to your actual Docker Hub username.

2.  **Login to Docker Hub**:
    ```bash
    docker login
    ```

3.  **Build and Push the Image**:
    ```bash
    make push
    ```
    This command builds the image and pushes it to `your-docker-hub-username/blossombot:latest`.

4.  **Update Deployment Manifest**:
    Open `k8s/deployment.yaml` and change the `image` field to point to your new Docker Hub image (e.g., `your-docker-hub-username/blossombot:latest`).

5.  **Configure for your Cluster**:
    -   You may want to change the service type in `k8s/service.yaml` to `LoadBalancer` if your cloud provider supports it, or configure a cloud-native Ingress.
    -   Ensure your `kubectl` context is pointing to your production cluster.

6.  **Deploy**:
    ```bash
    kubectl apply -f k8s/
    ```

## Running Locally (Without Kubernetes)

To run the application locally without Kubernetes, you'll need Python 3.9+ and Docker installed.

1.  **Set up Environment Variables**:
    Create a `.env` file from the `.env.example` template and fill in your Slack API credentials, specifically:
    -   `SLACK_BOT_TOKEN`: Your bot's OAuth token (starts with `xoxb-`).
    -   `SLACK_SIGNING_SECRET`: Your app's signing secret.
    -   `REDIS_HOST`: (e.g., `localhost` if running Redis locally).
    ```bash
    cp .env.example .env
    ```
2.  **Install Dependencies**: `pip install -r app/requirements.txt`
3.  **Start Redis**: `docker run -d -p 6379:6379 --name blossombot-redis redis:alpine`
4.  **Run the Application**: `uvicorn app.main:app --reload`
5.  **Expose to Slack (using ngrok)**: To receive events from Slack, your local application needs to be accessible from the internet. You can use `ngrok` for this.
    *   Download and install `ngrok`.
    *   Run `ngrok http 8000` (assuming your FastAPI app runs on port 8000 by default, check `uvicorn` output).
    *   Copy the `https` forwarding URL provided by ngrok.
    *   Configure this URL in your Slack App settings under "Event Subscriptions" and "Interactivity & Shortcuts" by appending `/slack/events` (e.g., `https://<ngrok_url>.ngrok.io/slack/events`).
