# RAG LangChain Setup

## Quick Start

```bash
# 1. Increase Colima memory (current: 2GB → recommended: 4GB+)
colima stop
colima start --memory 4 --cpu 2

# 2. Start Ollama with docker-compose
docker-compose up -d

# 3. Pull models
docker exec ollama-server ollama pull tinyllama
docker exec ollama-server ollama pull nomic-embed-text

# 4. Run the application
yarn dev
```

## Memory Issue

**Problem:** Colima has only 2GB memory, insufficient for running multiple Ollama models.

**Solution:** Restart Colima with 4-6GB:
```bash
colima stop
colima start --memory 6 --cpu 4  # For better performance
```

## Commands

```bash
# Build and run
yarn build          # Compile TypeScript
yarn start          # Run compiled code
yarn dev            # Build + run

# Check memory
colima list
docker stats ollama-server --no-stream

# Unload models (if needed)
curl -X POST http://localhost:11434/api/generate -d '{"model": "tinyllama", "keep_alive": 0}'
```

## Troubleshooting

Memory errors? Try:
1. Unload models before running
2. Reduce context size in `index.ts` (line 81)
3. Retrieve fewer documents (line 71)

