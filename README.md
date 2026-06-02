# 🌸 BlossomBot: High-Resilience AI Flower Shop

BlossomBot is a production-ready, AI-powered e-commerce chatbot designed for modern flower shops. It features an advanced RAG (Retrieval-Augmented Generation) engine, persistent state management, and a high-availability architecture built on Kubernetes.

## 🚀 Key Advanced Features (Production-Ready)

- **SOTA RAG Engine:** Hybrid search (Vector + Metadata) with dynamic score-based filter relaxation and Cross-Encoder reranking.
- **Sequential AI Task Queue (QoS):** Handles Gemini API rate limits (429 errors) by queuing requests and processing them sequentially with controlled delays.
- **Atomic User-Level Sequencing:** Uses Redis locks to guarantee that messages from the same user are processed in strict chronological order, even under extreme system load.
- **Empathetic Fallbacks:** Intelligent price-floor detection and semantic jitter to guide users when catalog constraints aren't met.
- **Automated Stress Testing:** In-cluster simulation of 20+ concurrent users with dedicated RabbitMQ telemetry isolation.
- **Microservices Architecture:** Scalable deployment with Redis (State), RabbitMQ (Messaging), and Qdrant (Vector DB).

## 📂 Technical Documentation

The project includes 18 comprehensive technical chapters located in the `documentation/` folder:
01. Chatbot Logical Flow
...
15. Quality of Service & Concurrency
16. Data Isolation & Test Strategy
17. Conversational UX & Fallbacks
18. Technical Standards & Code Quality

## 🛠 Quick Start (Production Validation)

```bash
# 1. Build and Deploy
make build-and-push-docker-images
make restart-chat

# 2. Run High-Concurrency Resilience Test
make stress-test

# 3. Verify Telemetry (Stress Queue)
make view-stress-queue
```

---
*Developed as a high-resilience AI E-commerce Gateway prototype.*
