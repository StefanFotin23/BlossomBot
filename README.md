# 🌸 BlossomBot: High-Resilience AI Flower Shop

BlossomBot is a production-ready, AI-powered e-commerce chatbot designed for modern flower shops. It features an advanced RAG (Retrieval-Augmented Generation) engine, persistent state management, and a high-availability architecture built on Kubernetes.

## 🚀 Key Advanced Features (Production-Ready)

- **SOTA RAG Engine:** Hybrid search (Vector + Metadata) with dynamic score-based filter relaxation and Cross-Encoder reranking.
- **Sequential AI Task Queue (QoS):** Handles Gemini API rate limits (429 errors) by queuing requests and processing them sequentially with controlled delays.
- **Atomic User-Level Sequencing:** Uses Redis locks to guarantee that messages from the same user are processed in strict chronological order, even under extreme load.
- **Multi-Intent Decomposition:** Breaks down complex user messages (e.g., "Add this AND search for that") into atomic tasks and executes them sequentially.
- **Deterministic Fast-Paths:** Instant processing of common keywords and jargon (Romanian/English) to offload AI tasks and provide zero-latency responses.
- **Empathetic Fallbacks:** Intelligent price-floor detection and semantic jitter to guide users when catalog constraints aren't met.
- **Automated Stress Testing:** In-cluster simulation of 20+ concurrent users with dedicated RabbitMQ telemetry isolation.
- **Microservices Architecture:** Scalable deployment with Redis (State), RabbitMQ (Messaging), and Qdrant (Vector DB).

## 📂 Strategic & Technical Documentation

The project includes 23 comprehensive chapters covering Technical, Business, and Marketing perspectives:
- **Core Logic:** 01. Logical Flow, 02. RAG Science, 05. Prompt Engineering.
- **Infrastructure:** 03. Architecture, 04. Deployment, 07. Message Brokers, 08. Redis State.
- **Resilience:** 09. Error Handling, 15. Quality of Service, 19. Multi-Intent Decomposition.
- **QA & Testing:** 14. Validation, 16. Data Isolation & Stress Testing.
- **Business & Strategy:** 
    - 21. **Business Value & Marketing ROI**
    - 22. **Technical Leadership & Deployment Agility**
    - 23. **Competitive Analysis (BlossomBot vs. FAN Courier)**

## 🛠 Quick Start (Production Validation)

```bash
# 1. Build and Deploy (Unified command)
make build-and-push-docker-images
make restart-chat

# 2. Run High-Concurrency Resilience Test
make stress-test

# 3. Verify Telemetry (Stress Queue)
make view-stress-queue
```

---
*Developed as a high-resilience AI E-commerce Gateway prototype. Proving that strategic engineering outperforms massive corporate budgets.*
