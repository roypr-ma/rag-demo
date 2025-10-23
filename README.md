# RAG LangChain Project

A progressive series of Retrieval-Augmented Generation (RAG) implementations using LangChain, LangGraph, and Ollama.

> **Tutorial-Based Learning:** This project implements official LangChain tutorials with local Ollama models instead of OpenAI.

## 📚 Four Parts

| Part | Description | Key Features | Tutorial |
|------|-------------|--------------|----------|
| **[Part 1: Basic RAG](1-basic-rag/)** | Foundation RAG pipeline | Simple retrieval + generation | [Tutorial](https://js.langchain.com/docs/tutorials/rag/) |
| **[Part 2: Conversational RAG](2-chat-history/)** | Adds chat history (2 approaches) | **Chains**: Fixed 1 retrieval<br>**Agents**: Multiple retrievals | [Tutorial](https://js.langchain.com/docs/tutorials/qa_chat_history) |
| **[Part 3: Agentic RAG](3-agentic-rag/)** | **ReAct framework** agent | Reason → Act → Observe → Learn | [Tutorial](https://docs.langchain.com/oss/javascript/langgraph/agentic-rag) |
| **[Part 4: Multi-Model Hybrid Search](4-hybrid-search/)** | **3 search types in 1 query** | **BM25** keyword + **Vector** semantic + **Graph** traversal with RRF fusion | Custom Implementation |

## 🛠️ Prerequisites

- **Node.js**: v22+
- **Yarn**: Package manager
- **Colima**: Docker runtime with 8GB+ memory

## 🚀 Quick Start

```bash
# 1. Ensure Colima has sufficient memory
# 8GB minimum (for Part 1, 2)
# 12GB recommended (if running Part 3 with llama3.1)
colima stop && colima start --memory 8 --cpu 4

# 2. Install dependencies
yarn install

# 3. Start services (Ollama + ArangoDB)
docker-compose up -d

# 4. Pull required models
docker exec ollama-server ollama pull llama2
docker exec ollama-server ollama pull nomic-embed-text

# For Part 2B (Agents)
docker exec ollama-server ollama pull qwen2.5:3b

# For Part 3 (Agentic RAG) - better tool-calling and instruction following
docker exec ollama-server ollama pull llama3.1

# 5. Build
yarn build

# 6. Run any part
yarn start:basic                        # Part 1: Basic RAG
yarn start:chat                         # Part 2: Conversational RAG (Chains)
yarn start:chat:agents                  # Part 2: Conversational RAG (Agents)
yarn start:agentic                      # Part 3: Agentic RAG
yarn start:hybrid "help building search with neural embeddings"  # Part 4: Hybrid search
yarn start:hybrid reset                 # Part 4: Reset database
```

## 📖 Project Structure

```
rag-langchain/
├── 1-basic-rag/
│   ├── index.ts              # Basic RAG implementation
│   └── README.md             # Detailed Part 1 docs
├── 2-chat-history/
│   ├── index-chains.ts       # Conversational RAG (Chains approach)
│   ├── index-agents.ts       # Conversational RAG (Agents approach)
│   └── README.md             # Detailed Part 2 docs
├── 3-agentic-rag/
│   ├── index.ts              # Agentic RAG with LangGraph
│   └── README.md             # Detailed Part 3 docs
├── 4-hybrid-search/
│   ├── index.ts              # Hybrid search with ArangoDB
│   └── README.md             # Detailed Part 4 docs
├── utils/
│   └── logger.ts             # Logging utilities
├── docker-compose.yml        # Ollama + ArangoDB services
├── package.json
└── README.md                 # This file
```

## 🎯 Learning Path

### Part 1: Basic RAG
**Start here** to understand RAG fundamentals.

```bash
yarn start:basic  # ~60-80s
```

**What you'll learn:**
- Document loading and chunking
- Vector embeddings and similarity search
- Basic retrieval-augmented generation
- LangGraph state management

📄 **[Read detailed Part 1 documentation →](1-basic-rag/README.md)**

---

### Part 2: Conversational RAG
**Add conversational memory** to handle follow-up questions. Implements **both approaches** from the tutorial:

#### Approach A: Chains (Predictable)
```bash
yarn start:chat  # ~180-240s (3 turns)
```
- **Fixed execution**: Exactly ONE retrieval per question
- **Predictable**: Same flow every time
- **Fast**: ~60-80s per question

#### Approach B: Agents (Flexible)
```bash
yarn start:chat:agents  # ~200-300s (varies)
```
- **Dynamic execution**: 0 to N retrievals per question
- **Adaptive**: Agent decides when it has enough info
- **Thorough**: Gathers multiple pieces of information
- **Requires**: Model with tool-calling support (qwen2.5, llama3.1, mistral)

**What you'll learn:**
- Chat history management
- History-aware retrieval
- Question reformulation
- Chain composition (Approach A)
- Agent loops and decision-making (Approach B)

**Example:**
```
Q1: "What is Task Decomposition?"
Q2: "What are common ways of doing it?"  ← Understands "it" = task decomposition
Q3: "Compare the approaches"            ← Agent may retrieve multiple times
```

📄 **[Read detailed Part 2 documentation →](2-chat-history/README.md)**

---

### Part 3: Agentic RAG (ReAct Framework)
**Build a ReAct agent** that reasons, acts, observes, and learns.

**⚠️ Prerequisites:** Pull the required model first:
```bash
docker exec ollama-server ollama pull llama3.1  # ~4.7GB
```

**Run:**
```bash
yarn start:agentic  # ~90-180s
```

- **ReAct framework** (Reasoning + Acting pattern)
- Document relevance grading after retrieval
- Query rewriting based on observations
- Self-correction through continuous evaluation
- **Requires**: `llama3.1` for better tool-calling and instruction following

**The ReAct Cycle:**
1. **Reason**: "Should I retrieve information?"
2. **Act**: Execute retrieval tool
3. **Observe**: Grade document quality
4. **Learn**: Rewrite query if needed, try again

**Why ReAct?** Unlike Part 2B (simple agent), Part 3 validates retrieved documents and self-corrects with query rewriting.

📄 **[Read detailed Part 3 documentation →](3-agentic-rag/README.md)**

---

### Part 4: Multi-Model Hybrid Search with ArangoDB
**Search a social network knowledge base** using 3 types of search in one query.

```bash
# Try this query that showcases all 3 search types:
yarn start:hybrid "help building search with neural embeddings"

# What happens:
# - BM25 finds "neural" + "embeddings" (exact keywords)
# - Vector understands search/ML expertise (semantic)
# - Graph finds collaborators (relationships)
# Result: Emma (12 yrs ⭐ Expert) + 4 team members!

yarn start:hybrid reset  # Reset database
```

**Three Search Types Combined:**
1. **BM25 keyword search** - Traditional full-text (exact terms)
2. **Vector semantic search** - AI embeddings (meaning & context)
3. **Graph traversal** - Follows relationships to find connected people

**Why Multi-Model?**
- **BM25 alone**: Finds people with exact keyword matches ✓
- **Vector alone**: Finds people with semantically similar expertise ✓
- **Graph traversal**: Discovers colleagues and collaborators ✓
- **Result**: 1 direct match becomes 5+ relevant people!

**What you'll learn:**
- BM25 keyword search with ArangoSearch
- Vector similarity search with embeddings
- Reciprocal Rank Fusion (RRF) to combine results
- Graph traversal for relationship-based discovery
- AQL multi-model queries
- Social network knowledge base architecture

**Real Example:**
```
Query: "help building search with neural embeddings"

❌ BM25 only:   1 person  (Emma - exact match)
❌ Vector only:  2 people (Alice, Henry - miss exact match!)
✅ Hybrid:       3 people (Emma ⭐ 12yrs #1, Alice 🔹 8yrs, Henry ⭐ 15yrs)
✅ + Graph:      5 people (+ Carol 🔹 5yrs, Bob 6yrs)

Result: Complete team with expertise levels. Emma is your top expert!
```

📄 **[Read detailed Part 4 documentation →](4-hybrid-search/README.md)**

---

## 🔧 Configuration

### Models

**Current Configuration:**
- **LLM (Part 1, 2A)**: `llama2` (~3.8GB) - Good reasoning, balanced performance
- **LLM (Part 2B)**: `qwen2.5:3b` (~2GB) - Tool-calling support
- **LLM (Part 3)**: `llama3.1` (~4.7GB, requires 8GB RAM) - Better tool-calling and instruction following
- **Embeddings (All Parts)**: `nomic-embed-text` (~274MB) - 768-dimensional vectors
- **Database (Part 4)**: `ArangoDB` - Multi-model database with BM25 and vector search

**💡 Note:** Part 3 uses a larger model (`llama3.1`) for more reliable document grading and query rewriting.

### Alternative Models

**For faster testing:**
```bash
docker exec ollama-server ollama pull tinyllama  # ~637MB, much faster
```

**For better quality:**
```bash
docker exec ollama-server ollama pull mistral  # ~4.1GB, better reasoning
```

Then update the model name in the respective `index.ts` file:
```typescript
const llm = new ChatOllama({
  model: "mistral",  // or "tinyllama"
  // ...
});
```

Browse models: [ollama.com/library](https://ollama.com/library)

## 📊 Feature Comparison

| Feature | Part 1 | Part 2 | Part 3 | Part 4 |
|---------|:------:|:------:|:------:|:------:|
| **Vector Search** | ✅ | ✅ | ✅ | ✅ |
| **Keyword Search (BM25)** | ❌ | ❌ | ❌ | ✅ |
| **Hybrid Search (RRF)** | ❌ | ❌ | ❌ | ✅ |
| **Chat History** | ❌ | ✅ | ❌ | ❌ |
| **Question Reformulation** | ❌ | ✅ | ✅ | ❌ |
| **Decision Making** | ❌ | ❌ | ✅ | ❌ |
| **Document Grading** | ❌ | ❌ | ✅ | ❌ |
| **Self-Correction** | ❌ | ❌ | ✅ | ❌ |
| **LLM Generation** | ✅ | ✅ | ✅ | ❌ |
| **Persistent Storage** | ❌ | ❌ | ❌ | ✅ |
| **Graph Capabilities** | ❌ | ❌ | ❌ | ✅ |

## 🐛 Troubleshooting

### "model runner has unexpectedly stopped"

**Cause:** Insufficient memory for llama2.

**Solution:**
```bash
colima list  # Check current memory
colima stop && colima start --memory 8 --cpu 4
docker-compose restart
```

### Slow Performance

**Solutions:**
1. Use `tinyllama` for testing (much faster)
2. Reduce chunk size in text splitters
3. Allocate more CPU/memory to Colima

### Memory Issues During Long Runs

Unload models between runs:
```bash
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "llama2", "keep_alive": 0}'
```

## 💡 Tech Stack

### Framework & Orchestration
- **[LangChain.js](https://js.langchain.com/)** - LLM application framework (chains, agents, retrievers)
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Graph-based workflow orchestration for agentic patterns

### LLM Infrastructure
- **[Ollama](https://ollama.com/)** - Local LLM runtime (via Docker)
- **[ArangoDB](https://www.arangodb.com/)** - Multi-model database (via Docker)

### Models
- **[Llama2](https://ollama.com/library/llama2)** (~3.8GB) - Part 1, 2A: General-purpose reasoning
- **[Qwen2.5:3b](https://ollama.com/library/qwen2.5)** (~2GB) - Part 2B: Lightweight tool-calling
- **[Llama3.1](https://ollama.com/library/llama3.1)** (~4.7GB) - Part 3: Advanced tool-calling and instruction following
- **[Nomic Embed Text](https://ollama.com/library/nomic-embed-text)** (~274MB) - All parts: 768-dimensional embeddings

## 📚 Resources

### Official Tutorials
- [LangChain RAG Tutorial](https://js.langchain.com/docs/tutorials/rag/)
- [Q&A with Chat History](https://js.langchain.com/docs/tutorials/qa_chat_history)
- [Agentic RAG with LangGraph](https://docs.langchain.com/oss/javascript/langgraph/agentic-rag)

### Documentation
- [LangChain JS Docs](https://js.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Model Library](https://ollama.com/library)

## 📝 License

MIT

---

**Ready to start?** Begin with [Part 1: Basic RAG →](1-basic-rag/README.md)
