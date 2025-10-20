# RAG LangChain Project

A progressive series of Retrieval-Augmented Generation (RAG) implementations using LangChain, LangGraph, and Ollama.

> **Tutorial-Based Learning:** This project implements official LangChain tutorials with local Ollama models instead of OpenAI.

## 📚 Three Parts

| Part | Description | Key Features | Tutorial |
|------|-------------|--------------|----------|
| **[Part 1: Basic RAG](1-basic-rag/)** | Foundation RAG pipeline | Simple retrieval + generation | [Tutorial](https://js.langchain.com/docs/tutorials/rag/) |
| **[Part 2: Conversational RAG](2-chat-history/)** | Adds chat history | Context-aware retrieval | [Tutorial](https://js.langchain.com/docs/tutorials/qa_chat_history) |
| **[Part 3: Agentic RAG](3-agentic-rag/)** | Intelligent agent | Decision-making + self-correction | [Tutorial](https://docs.langchain.com/oss/javascript/langgraph/agentic-rag) |

## 🚀 Quick Start

```bash
# 1. Ensure Colima has sufficient memory (8GB for llama2)
colima stop && colima start --memory 8 --cpu 4

# 2. Install dependencies
yarn install

# 3. Start Ollama
docker-compose up -d

# 4. Pull required models
docker exec ollama-server ollama pull llama2
docker exec ollama-server ollama pull nomic-embed-text

# 5. Build
yarn build

# 6. Run any part
yarn start:basic        # Part 1: Basic RAG
yarn start:chat         # Part 2: Conversational RAG
yarn start:agentic      # Part 3: Agentic RAG
```

## 📖 Project Structure

```
rag-langchain/
├── 1-basic-rag/
│   ├── index.ts              # Basic RAG implementation
│   └── README.md             # Detailed Part 1 docs
├── 2-chat-history/
│   ├── index.ts              # Conversational RAG with chains
│   └── README.md             # Detailed Part 2 docs
├── 3-agentic-rag/
│   ├── index.ts              # Agentic RAG with LangGraph
│   └── README.md             # Detailed Part 3 docs
├── docker-compose.yml        # Ollama service
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
**Add conversational memory** to handle follow-up questions.

```bash
yarn start:chat  # ~180-240s (3 turns)
```

**What you'll learn:**
- Chat history management
- History-aware retrieval
- Question reformulation
- Chain composition

**Example:**
```
Q1: "What is Task Decomposition?"
Q2: "What are common ways of doing it?"  ← Understands "it" = task decomposition
Q3: "Can you give me specific examples?" ← Maintains full context
```

📄 **[Read detailed Part 2 documentation →](2-chat-history/README.md)**

---

### Part 3: Agentic RAG
**Build an intelligent agent** that makes decisions.

```bash
yarn start:agentic  # ~90-180s
```

**What you'll learn:**
- Tool-based architecture
- Conditional graph execution
- Document relevance grading
- Self-correction loops
- Decision-making agents

**Key capabilities:**
- Decides when to retrieve vs respond directly
- Validates document relevance before answering
- Rewrites queries if documents aren't relevant

📄 **[Read detailed Part 3 documentation →](3-agentic-rag/README.md)**

---

## 🔧 Configuration

### Models

**Current Configuration:**
- **LLM**: `llama2` (~3.8GB) - Good reasoning, balanced performance
- **Embeddings**: `nomic-embed-text` (~274MB) - 768-dimensional vectors

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

## 🛠️ Prerequisites

- **Node.js**: v22+
- **Yarn**: Package manager
- **Colima**: Docker runtime with 8GB+ memory
- **Docker**: For running Ollama

## 📊 Feature Comparison

| Feature | Part 1 | Part 2 | Part 3 |
|---------|:------:|:------:|:------:|
| **Basic Retrieval** | ✅ | ✅ | ✅ |
| **Chat History** | ❌ | ✅ | ❌ |
| **Question Reformulation** | ❌ | ✅ | ✅ |
| **Decision Making** | ❌ | ❌ | ✅ |
| **Document Grading** | ❌ | ❌ | ✅ |
| **Self-Correction** | ❌ | ❌ | ✅ |
| **Multi-Document Search** | ❌ | ❌ | ✅ |
| **Conditional Logic** | ❌ | ❌ | ✅ |

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

- **[LangChain.js](https://js.langchain.com/)** - LLM application framework
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Graph-based workflow orchestration
- **[Ollama](https://ollama.com/)** - Local LLM runtime
- **[Llama2](https://ollama.com/library/llama2)** - Meta's open-source LLM
- **[Cheerio](https://cheerio.js.org/)** - Web scraping

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
