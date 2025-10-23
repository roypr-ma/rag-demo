# Part 4: Multi-Model Hybrid Search with ArangoDB

A **social network knowledge base** demonstrating true multi-model hybrid search by combining **3 types of search** in one query using **ArangoDB**.

## What is Hybrid Search?

**Hybrid search** means combining multiple search techniques to get better results than any single approach:

### The 3 Search Types Used:

#### 1. **BM25 Keyword Search** (Traditional IR)
- **What**: Classic full-text search using TF-IDF and document length normalization
- **When**: Finds exact keyword matches like "graph database" or "real-time messaging"
- **Strength**: Precision - finds documents with specific terms
- **Limitation**: Misses semantically similar content with different wording

#### 2. **Vector Similarity Search** (Semantic AI)
- **What**: Converts text to embeddings (768-dim vectors) and finds nearest neighbors
- **When**: Finds conceptually similar content even without exact keywords
- **Strength**: Recall - understands meaning and context
- **Limitation**: May miss exact technical terms or names

#### 3. **Graph Traversal** (Relationship-based)
- **What**: Follows edges in a graph to find connected documents
- **When**: Discovers related articles through their relationships
- **Strength**: Context expansion - finds relevant content via connections
- **Limitation**: Depends on quality of relationship data

### Why Combine All Three?

Let's use a realistic query to see why you need all three search types:

**User Query**: `"I need help building a search feature with neural embeddings"`

This query is perfect because it combines specific technical terms with natural language - just like real users search!

| What Each Search Type Finds | Result | Why It Happens |
|------------------------------|--------|----------------|
| **❌ BM25 only** | Emma (1 person) | Finds "neural" + "embeddings" but MISSES "semantic search" experts |
| **❌ Vector only** | Alice, Henry (2 people) | Understands concept but MISSES Emma who literally says "neural embeddings"! |
| **✅ Hybrid (BM25+Vector)** | Emma ⭐, Alice 🔹, Henry ⭐ (3 people) | RRF fusion ranks best matches. **Emma = 12 years expert!** |
| **✅✅ + Graph Traversal** | + Carol 🔹, Bob (5 people) | Discovers the COMPLETE team including collaborators |

**Why This Shows Hybrid's Power:**
- Each search type has a blind spot when used alone
- BM25 misses semantic similarity (search ≈ neural embeddings)
- Vector misses exact keyword matches (Emma literally has the skills!)
- Graph discovers the actual team you need to build the feature
- **Shows expertise levels**: ⭐ Expert, 🔹 Senior, and years of experience

**Result**: 1 person → 5 people with **Emma (12 yrs, Expert)** as the top expert!

## Architecture

### System Components

```mermaid
graph TB
    subgraph "User Layer"
        Q[🔍 Search Query<br/>"friend connections"]
    end
    
    subgraph "Application Layer"
        APP[📱 Search Application]
        EMB[🧠 Ollama Embeddings<br/>nomic-embed-text]
    end
    
    subgraph "ArangoDB Multi-Model Database"
        direction TB
        
        subgraph "Storage"
            DOCS[(📄 Document Collection<br/>Knowledge Base Articles<br/>+ Embeddings)]
            EDGES[(🔗 Edge Collection<br/>Article Relationships)]
        end
        
        subgraph "Indexes & Views"
            VIDX[🎯 Vector Index<br/>Cosine Similarity<br/>768 dimensions]
            BVIEW[📚 ArangoSearch View<br/>BM25 Scoring<br/>Full-text Index]
        end
        
        subgraph "Graph"
            GRAPH[🕸️ Named Graph<br/>docs_graph]
        end
        
        subgraph "Query Engine"
            AQL[⚙️ AQL Multi-Model Query]
        end
    end
    
    subgraph "Search Pipeline"
        direction LR
        S1[🔎 BM25<br/>Keyword Search]
        S2[🧠 Vector<br/>Semantic Search]
        S3[🔄 RRF<br/>Score Fusion]
        S4[🕸️ Graph<br/>Traversal]
        
        S1 --> S3
        S2 --> S3
        S3 --> S4
    end
    
    subgraph "Results"
        R1[🎯 Direct Matches<br/>3 articles]
        R2[🔗 Related via Graph<br/>5+ articles]
        FINAL[📋 Ranked Results<br/>8+ articles total]
        
        R1 --> FINAL
        R2 --> FINAL
    end
    
    Q --> APP
    APP --> EMB
    EMB --> APP
    APP --> AQL
    
    DOCS --> VIDX
    DOCS --> BVIEW
    DOCS --> GRAPH
    EDGES --> GRAPH
    
    VIDX --> AQL
    BVIEW --> AQL
    GRAPH --> AQL
    
    AQL --> S1
    AQL --> S2
    S4 --> R1
    S4 --> R2
    
    FINAL --> APP
    APP --> Q
    
    style Q fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style EMB fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style DOCS fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style EDGES fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style VIDX fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style BVIEW fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style GRAPH fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style AQL fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    style S3 fill:#fff9c4,stroke:#f9a825,stroke-width:2px
    style FINAL fill:#c8e6c9,stroke:#388e3c,stroke-width:3px
```

**Key Components:**

1. **Document Collection**: Stores knowledge base articles with text + 768-dim embeddings
2. **Edge Collection**: Stores relationships (enables, powers, uses, etc.)
3. **Vector Index**: Enables fast similarity search on embeddings
4. **ArangoSearch View**: Provides BM25 full-text search
5. **Named Graph**: Defines traversable relationships between articles
6. **AQL Engine**: Executes multi-model query combining all search types

### Why Hybrid Search?

| Search Type | Strengths | Weaknesses | Example |
|-------------|-----------|------------|---------|
| **Keyword-only** | Exact matches | Misses synonyms/concepts | Finds "graph" but misses "connected data" |
| **Vector-only** | Semantic understanding | Misses exact keywords | Finds concepts but misses exact terms |
| **Hybrid (BM25+Vector)** | Precision + recall | Limited to direct matches | Combines both approaches |
| **Multi-Model (Hybrid+Graph)** | Everything above + relationships | Most complex | Finds direct matches AND related docs |

**Real Example from This Implementation:**

Query: `"I need help building a search feature with neural embeddings"`

- **BM25**: Finds Emma (12 yrs ⭐ Expert) - has "neural" + "embeddings"
- **Vector**: Finds Alice (8 yrs 🔹 Senior), Henry (15 yrs ⭐ Expert)
- **Hybrid**: RRF ranks Emma #1 - **Principal Research Scientist with 12 years!**
- **Graph**: Adds Carol (5 yrs), Bob (6 yrs) - the supporting team
- **Result**: Complete team with clear expertise levels - you know who's the expert!

## Running Part 4

### Prerequisites

```bash
# Start services
docker-compose up -d

# Pull embedding model (one-time)
docker exec -it ollama-server ollama pull nomic-embed-text  # ~274MB
```

### Run Searches

Just run your search! The first time, it will automatically set up the database:

```bash
# Search the social network knowledge base
yarn start:hybrid "help building search with neural embeddings"

# The query above demonstrates all 3 search types:
# - BM25 finds exact keyword matches ("neural", "embeddings")
# - Vector understands semantic meaning (search systems, ML)
# - Graph discovers collaborators (people who work together)
# - Shows expertise: ⭐ Expert, 🔹 Senior + years of experience

# Reset database (drop and recreate on next search)
yarn start:hybrid reset
```

**What happens:**
1. **BM25** finds people with exact keywords in their profiles
2. **Vector search** finds people with semantically similar expertise
3. **RRF fusion** combines and ranks both result sets
4. **Graph expansion** automatically includes connected colleagues

The database setup happens automatically on first run and includes:
- Creating database, collections, and graph
- Ingesting 10 professional profiles with embeddings
- Creating 18 relationships showing how people collaborate

**Reset Database:**
Use `yarn start:hybrid reset` to drop the database and start fresh. Useful for:
- Testing setup process
- Clearing all data
- Troubleshooting issues

## Reciprocal Rank Fusion (RRF)

**Why RRF?** BM25 scores (0-15) and cosine similarity (0-1) use different scales. RRF converts ranks to comparable scores.

**Formula:** `score = 1 / (k + rank)` where k=60 (constant)

**Example from Social Network KB:**

Query: "help building search with neural embeddings"

**Two ranked lists:**
- **BM25**: [emma (rank=1, "neural embeddings"), alice (rank=3, "embeddings")]
- **Vector**: [alice (rank=1, search/ML expert), emma (rank=2, neural networks), henry (rank=3, search systems)]

**RRF Calculation:**
```
alice:  1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323 ✅ Highest!
emma:   1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325 ✅ Even Higher!
henry:  1/(60+3)            = 0.0159
```

**Final Ranking:** emma (rank 1+2) > alice (rank 3+1) > henry (rank 3 only)

**Why it matters:** Emma appears in BOTH lists with good ranks, so RRF boosts her to the top!
**Then Graph adds:** Carol (works with Alice), Bob (integrates features) → 5 total people

**Key Benefits:**
- People appearing in BOTH lists get boosted (alice, henry)
- No score normalization needed
- Simple and effective

## The Three Search Technologies Explained

### Search Type Comparison

| Search Type | Technology | Use Case | Example Query Result |
|-------------|-----------|----------|---------------------|
| **BM25 Keyword** | TF-IDF + doc length normalization | Exact term matching | "graph database" → finds Bob (mentions "graph databases" exactly) |
| **Vector Semantic** | Neural embeddings + cosine similarity | Conceptual similarity | "graph database" → finds Frank (DBA, related concept) |
| **Graph Traversal** | Edge following + relationship types | Related people discovery | From Bob → finds Henry (reports to), David (collaborates with) |

### Why This Matters

**Real User Query:**

Query: "I need help building a search feature with neural embeddings"

**Without Hybrid (BM25 only):**
- Finds Emma (says "neural" + "embeddings") ✓
- MISSES Alice (says "semantic search" not exact keywords) ❌
- **Result**: 1 person (but you don't know if she's the best)

**With Hybrid (BM25 + Vector):**
- BM25: Finds Emma (exact match) - **12 years ⭐ Expert**
- Vector: Finds Alice (8 yrs 🔹 Senior), Henry (15 yrs ⭐ Expert)
- RRF: Ranks Emma #1 (she's the neural embeddings expert!)
- **Result**: 3 people with **clear expertise levels**

**With Graph (+Traversal):**
- Discovers Carol (5 yrs 🔹 - works with Alice on ranking systems)
- Discovers Bob (6 yrs - backend developer who can integrate the feature)
- **Result**: Complete team hierarchy - Emma leads, Alice/Henry support, Carol/Bob assist

## Implementation

### Key Components

**Document Structure:**
```typescript
{
  _key: "doc1",
  text: "ArangoDB is a multi-model database...",
  embedding: [0.23, -0.15, 0.42, ...] // 768-dim vector
}
```

**Indexes:**
- **Vector Index**: For semantic similarity search
- **ArangoSearch View**: For BM25 keyword search

### AQL Query (Simplified)

```aql
// 1. BM25 Keyword Search
FOR doc IN docs_view
  SEARCH ANALYZER(doc.text IN TOKENS(query, 'text_en'), 'text_en')
  SORT BM25(doc) DESC
  LIMIT 5

// 2. Vector Similarity Search
FOR doc IN docs
  LET similarity = COSINE_SIMILARITY(doc.embedding, query_vector)
  SORT similarity DESC
  LIMIT 5

// 3. Apply RRF and combine
FOR doc IN UNION(bm25_rrf, vector_rrf)
  COLLECT _id = doc._id
  AGGREGATE final_score = SUM(doc.rrf_score)
  SORT final_score DESC
```

## Configuration

**Models:**
- **Embeddings**: `nomic-embed-text` (~274MB, 768 dimensions)
- No LLM required (pure search, not RAG generation)

**Database:**
- **ArangoDB**: Version 3.11+
- **Database**: `hybrid_search_db`
- **Collection**: `docs`
- **View**: `docs_view`

**Parameters:**
- **RRF Constant (k)**: 60
- **Results**: 5 per search type
- **Similarity**: Cosine
- **Analyzer**: `text_en`

## Comparison

| Feature | Part 1 | Part 4 |
|---------|--------|--------|
| **Search Type** | Vector only | Hybrid (BM25 + Vector) |
| **Storage** | In-memory | Persistent |
| **Keyword Search** | ❌ | ✅ BM25 |
| **Result Fusion** | ❌ | ✅ RRF |
| **LLM Generation** | ✅ | ❌ (search only) |

## Use Cases

✅ **Use hybrid search when:**
- You need both precision (keywords) and recall (semantics)
- Users search with exact terms OR natural language
- You want the best search quality
- Building search engines, recommendation systems

❌ **Use vector-only when:**
- Pure semantic similarity is sufficient
- Simpler implementation preferred

## Advantages & Limitations

**Advantages:**
- ✅ Superior search quality (precision + recall)
- ✅ Robust to embedding quality
- ✅ Persistent storage
- ✅ Production-ready (ACID, clustering)

**Limitations:**
- ❌ More complex setup (ArangoDB + AQL)
- ❌ No LLM generation (search only)
- ❌ No chat history

## Extending

**Add to RAG Pipeline:**
```typescript
const hybridRetriever = async (query: string) => {
  const embedding = await getEmbedding(query);
  const results = await executeHybridSearch(query, embedding);
  return results.map(r => new Document({ pageContent: r.text }));
};
```

**Add Features:**
- Chat history storage in ArangoDB
- Metadata filtering (date, category, etc.)
- Reranking with cross-encoders
- Multiple graph traversal depths

## Troubleshooting

### Reset Database

If you encounter issues or want to start fresh:

```bash
yarn start:hybrid reset
```

This drops the entire database. The next search will automatically recreate everything.

### Common Issues

**"Database connection failed"**
- Ensure ArangoDB is running: `docker ps | grep arangodb`
- Restart if needed: `docker-compose restart arangodb`

**"Vector index feature not enabled"**
- Check `docker-compose.yml` has `--experimental-vector-index true`
- Restart containers: `docker-compose down && docker-compose up -d`

**"Model not found"**
- Pull the embedding model: `docker exec -it ollama-server ollama pull nomic-embed-text`

**Stale data or schema issues**
- Use reset: `yarn start:hybrid reset`
- Then run any search to recreate

## Resources

**ArangoDB:**
- [Documentation](https://www.arangodb.com/docs/)
- [AQL Query Language](https://www.arangodb.com/docs/stable/aql/)
- [ArangoSearch](https://www.arangodb.com/docs/stable/arangosearch.html)
- [Vector Search](https://www.arangodb.com/docs/stable/indexing-vector.html)

**Hybrid Search:**
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Hybrid Search Overview](https://www.pinecone.io/learn/hybrid-search-intro/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

---

**Ready to try?** Run `yarn start:hybrid "your search query"` →
