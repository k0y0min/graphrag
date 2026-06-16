<div align="center">
  <h1>GraphRAG</h1>
  <p><em>An Advanced Graph Retrieval-Augmented Generation System</em></p>

  <!-- Badges -->
  <p>
    <img alt="Python" src="https://img.shields.io/badge/Python-3.13+-blue.svg?logo=python&logoColor=white" />
    <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-005571?logo=fastapi" />
    <img alt="KuzuDB" src="https://img.shields.io/badge/Database-KuzuDB-orange" />
    <img alt="vLLM" src="https://img.shields.io/badge/Local_LLM-vLLM-purple" />
    <img alt="Gemini" src="https://img.shields.io/badge/API_LLM-Gemini_3_Flash-4285F4?logo=google" />
    <img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg" />
  </p>
  <!--
  <p>
    <img alt="Hits" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fyourusername%2Fgraphrag&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/>
  </p>
  -->
</div>

---

## 📖 Overview

GraphRAG is a modern, full-stack application that leverages the power of Knowledge Graphs and Large Language Models (LLMs) to ingest raw text, extract structured entities and relationships, and allow users to query their data both semantically and visually. 


### ✨ Key Features
*   **Perplexity-Based Chunking**: Instead of arbitrarily splitting text, it uses a local model (Gemma via vLLM) to calculate sentence perplexity, splitting text dynamically at semantic boundaries (topic shifts).
*   **Smart Entity Resolution**: Uses a combination of deterministic string similarity (`difflib`) and LLM-powered merging to prevent duplicate entities and consolidate descriptions.
*   **Hybrid LLM Architecture**: Uses local vLLM for heavy, repetitive tasks (chunking perplexity) and Gemini (via LiteLLM) for complex reasoning (extraction and resolution).
*   **Embedded Graph Database**: Utilizes KuzuDB for ultra-fast, local graph storage and Cypher querying.
*   **Premium UI**: A sleek, responsive frontend featuring a Nord-inspired color palette, glassmorphism components, and interactive 3D graph visualization.

---

## 🏗️ Architecture

The system is designed as a monolithic API serving a static frontend, communicating with specialized LLM services.

```mermaid
graph TD
    subgraph Frontend [Frontend Web App]
        UI[Glassmorphism UI]
        Vis[Interactive 3D Graph]
    end

    subgraph Backend [FastAPI Backend]
        API[API Endpoints]
        Pipe[GraphRAG Pipeline]
        Chunker[Perplexity Chunker]
        Extractor[Entity Extractor]
        Storage[KuzuDB Storage]
    end

    subgraph LLM_Services [LLM Services]
        vLLM[vLLM Local Server<br/>Gemma-3-12b]
        LiteLLM[LiteLLM Wrapper<br/>Gemini 3 Flash]
    end

    UI -->|Ingest/Query Requests| API
    API --> Pipe
    Pipe --> Chunker
    Pipe --> Extractor
    Pipe --> Storage

    Chunker -->|Calculate PPL| vLLM
    Extractor -->|Extract & Resolve| LiteLLM
    
    Storage -->|Return Graph| Vis
```
---

## 📋 TODO:

- [X] ~~Implement user authentication & session management properly.~~ (half decent)
- [ ] Add support for document/multimodal uploads.
- [ ] Fine-tune perplexity spike threshold for different types of texts.
- [ ] Add graph layout saving(freeze nodes in specific positions) and other QOL features. (import functionality from portfolio website?)
- [ ] Checkout FalkorDB and perhaps migrate.
- [ ] Improve entity resolution (long term).
---

## 📜 License Information

Licensed under the **MIT License**.
