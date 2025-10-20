# Part 3: Agentic RAG with LangGraph

An intelligent RAG agent that can make decisions, validate retrieved content, and self-correct.

## Overview

This implementation follows the [LangChain Agentic RAG Tutorial](https://docs.langchain.com/oss/javascript/langgraph/agentic-rag) and demonstrates advanced RAG capabilities using LangGraph's conditional execution flow.

## Key Features

### 🤔 Intelligent Decision Making
- The agent decides whether to retrieve documents or respond directly
- Simple greetings get immediate responses without retrieval
- Complex questions trigger the retrieval tool

### 📊 Document Grading
- Validates relevance of retrieved documents before generating answers
- Uses LLM to assess if documents contain information relevant to the question
- Only generates answers when documents are deemed relevant

### ✍️ Self-Correction Loop
- If retrieved documents aren't relevant, the agent rewrites the query
- Improved query is used to retrieve again
- Loops until relevant documents are found

### 🔧 Tool-Based Architecture
- Uses a retriever tool that the agent can invoke
- LangGraph's ToolNode handles tool execution
- Conditional edges route based on agent decisions

## Architecture

### Graph Structure

```
START
  ↓
generateQueryOrRespond ←──────┐
  ↓                           │
  ├─→ (no tools) → END        │
  └─→ (has tools) → retrieve  │
                      ↓       │
                 gradeDocuments│
                      ↓       │
                      ├─→ (relevant) → generate → END
                      └─→ (not relevant) → rewrite ───┘
```

### Nodes

1. **generateQueryOrRespond**: Agent decides whether to use retrieval or respond directly
2. **retrieve**: Custom node that executes the vector store retrieval
3. **gradeDocuments**: LLM grades document relevance
4. **rewrite**: Reformulates the question for better retrieval
5. **generate**: Creates final answer using relevant context

### Conditional Edges

- `shouldRetrieve()`: Routes to retrieval or direct response
- `checkRelevance()`: Routes to generation or query rewriting

## Data Sources

Loads and indexes three blog posts by Lilian Weng:
1. [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
2. [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
3. [Adversarial Attacks on LLMs](https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/)

## Running the Demo

```bash
# Make sure Ollama is running with llama2 and nomic-embed-text models
docker-compose up -d

# Build and run
yarn start:agentic
```

## Expected Output

The demo runs three test queries:

### Test 1: Complex Question (Requires Retrieval)
```
Question: "What are the types of memory in LLM agents?"

Flow:
  generateQueryOrRespond → retrieve → gradeDocuments → generate

Expected: Detailed answer about memory types in LLM agents
```

### Test 2: Simple Greeting (No Retrieval)
```
Question: "Hello! How are you?"

Flow:
  generateQueryOrRespond → END

Expected: Direct response without retrieval
```

### Test 3: Specific Question (Retrieval + Grading)
```
Question: "What does Lilian Weng say about chain of thought prompting?"

Flow:
  generateQueryOrRespond → retrieve → gradeDocuments → generate

Expected: Answer about chain of thought from the blog posts
```

## Key Differences from Part 2

| Feature | Part 2 (Chains) | Part 3 (Agentic) |
|---------|----------------|------------------|
| **Decision Making** | Always retrieves | Decides if retrieval needed |
| **Validation** | No relevance checking | Grades document relevance |
| **Self-Correction** | No query rewriting | Rewrites queries if needed |
| **Execution Flow** | Linear chain | Conditional graph with loops |
| **Chat History** | Yes (conversational) | No (single-turn) |
| **Tools** | No tools | Uses retriever tool |

## Technical Details

### Document Processing
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Retrieval**: Top 3 most similar documents

### Models
- **LLM**: llama2 (for agent reasoning, grading, rewriting, generation)
- **Embeddings**: nomic-embed-text (768-dimensional vectors)

### Prompts

**Document Grading Prompt**:
```
You are a grader assessing relevance of retrieved documents to a user question.
[Documents and question provided]
Respond with "yes" if relevant, "no" if not relevant.
```

**Query Rewriting Prompt**:
```
Look at the input and try to reason about the underlying semantic intent/meaning.
Formulate an improved question that is more specific and likely to retrieve relevant information.
```

**Answer Generation Prompt**:
```
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
Keep the answer concise (3 sentences max).
```

## Implementation Notes

### Custom Tool Implementation
Instead of using LangChain's built-in tool binding (which has deep TypeScript type issues), this implementation:
- Uses **prompt engineering** to instruct the LLM about tool availability
- Parses LLM responses for `TOOL_CALL: retrieve_blog_posts` pattern
- Implements a **custom retrieve node** instead of `ToolNode`
- Uses `instanceof` checks instead of type casting

This approach is more explicit, easier to debug, and avoids TypeScript type instantiation depth limits.

### State Management
Uses LangGraph's `Annotation.Root` for type-safe state:
```typescript
const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});
```

Messages accumulate throughout the graph execution, allowing nodes to access the full conversation history.

## Potential Enhancements

1. **Max Iterations**: Add a counter to prevent infinite rewriting loops
2. **Streaming**: Stream responses as they're generated
3. **Multi-tool Support**: Add more tools (web search, calculator, etc.)
4. **Conversation Memory**: Combine with Part 2's chat history
5. **Parallel Retrieval**: Retrieve from multiple sources simultaneously
6. **Confidence Scores**: Add numerical relevance scores instead of yes/no
7. **Human-in-the-Loop**: Allow human approval before tool execution

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Agentic RAG Tutorial](https://docs.langchain.com/oss/javascript/langgraph/agentic-rag)
- [LangGraph Conditional Edges](https://langchain-ai.github.io/langgraph/how-tos/branching/)
- [Tool Calling with LangChain](https://js.langchain.com/docs/modules/agents/tools/)

