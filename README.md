# RAG LangChain Project

A Retrieval-Augmented Generation (RAG) system using LangChain, LangGraph, and Ollama. Loads web content, creates embeddings, and answers questions using local LLMs.

## Prerequisites

- **Node.js**: v22+
- **Yarn**: Package manager
- **Colima**: Docker runtime with 4GB+ memory (6-8GB recommended)

## Quick Start

```bash
# 1. Ensure Colima has sufficient memory
colima list
colima stop && colima start --memory 6 --cpu 4

# 2. Install dependencies
yarn install

# 3. Start Ollama
docker-compose up -d

# 4. Pull models
docker exec ollama-server ollama pull tinyllama
docker exec ollama-server ollama pull nomic-embed-text

# 5. Build and run
yarn dev
```

## Usage

### Run the Application

```bash
# Build and run together
yarn dev

# Or separately
yarn build    # Compile TypeScript
yarn start    # Run compiled code
```

## How It Works

1. **Data Loading**: Scrapes web content using Cheerio (selects `<p>` tags by default)
2. **Text Splitting**: Divides content into 1000-character chunks with 200-character overlap
3. **Embedding**: Converts chunks to 768-dimensional vectors using `nomic-embed-text`
4. **Indexing**: Stores vectors in MemoryVectorStore for similarity search
5. **Retrieval**: Finds most relevant chunks based on question similarity
6. **Generation**: Uses LangChain RAG prompt + retrieved context to generate answer with `tinyllama`

## Configuration

### Models

- **LLM**: `tinyllama` (~637MB) - Fast, lightweight 1B parameter model
- **Embeddings**: `nomic-embed-text` (~274MB) - 768-dimensional embeddings

### Changing Models

```typescript
const llm = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "llama2",  // Options: llama2, mistral, mixtral, etc.
  temperature: 0,
});

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://localhost:11434",
  model: "all-minilm",  // Other embedding models
});
```

Available models: [ollama.com/library](https://ollama.com/library)

## Troubleshooting

### Error: "model runner has unexpectedly stopped"

**Problem:** Insufficient memory allocated to Colima. Running Ollama with multiple models requires at least 4GB (6-8GB recommended).

**Solution:**

```bash
# Check current memory
colima list

# Restart Colima with more memory
colima stop
colima start --memory 6 --cpu 4

# Restart Ollama
docker-compose down
docker-compose up -d
```

**Free up memory** (if needed):
```bash
# Unload models
curl -X POST http://localhost:11434/api/generate -d '{"model": "tinyllama", "keep_alive": 0}'
curl -X POST http://localhost:11434/api/generate -d '{"model": "nomic-embed-text", "keep_alive": 0}'
```

**Useful commands:**
```bash
# Check memory usage
docker stats ollama-server --no-stream

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check container logs
docker logs ollama-server
```

## Architecture

```
┌─────────────┐
│  Web Page   │
└──────┬──────┘
       │ Cheerio
       ▼
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │ Split
       ▼
┌─────────────┐      ┌──────────────┐
│   Chunks    │─────▶│  Embeddings  │
└─────────────┘      │ (Ollama API) │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Vector Store │
                     └──────┬───────┘
                            │
    ┌───────────────────────┴──────────┐
    │                                   │
    ▼                                   │
┌─────────────┐                         │
│  Question   │                         │
└──────┬──────┘                         │
       │ Embed                          │
       ▼                                │
┌─────────────┐                         │
│  Retrieval  │◀────────────────────────┘
└──────┬──────┘   Similarity Search
       │
       ▼
┌─────────────┐      ┌──────────────┐
│   Context   │─────▶│  LLM (TinyLlama)
└─────────────┘      └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │    Answer    │
                     └──────────────┘
```

## Limitations

- **In-memory storage**: Vector store is cleared on restart (no persistence)
- **Regeneration required**: Embeddings must be recreated each run
- **Model constraints**: `tinyllama` has limited reasoning - consider using `llama2` or `mistral` for better results
- **Context window**: Limited by model's maximum context length
- **Out-of-domain queries**: May produce nonsensical answers for questions outside the indexed content

## Tech Stack

- **[LangChain](https://js.langchain.com/)** - Framework for LLM applications
- **[LangGraph](https://langchain-ai.github.io/langgraph/)** - Graph-based workflow orchestration
- **[Ollama](https://ollama.com/)** - Local LLM runtime
- **[Cheerio](https://cheerio.js.org/)** - Web scraping

## Resources

- [LangChain JS Documentation](https://js.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Model Library](https://ollama.com/library)
- [RAG Concepts Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Colima Documentation](https://github.com/abiosoft/colima)

## License

MIT