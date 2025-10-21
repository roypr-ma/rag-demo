import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { HumanMessage, AIMessage, BaseMessage, isAIMessage, isToolMessage } from "@langchain/core/messages";
import { logSection, logQuestion, logDivider, logTime, logSeparator, logSummary } from "../utils/logger.js";

// ============================================================================
// PART 2B: CONVERSATIONAL RAG WITH AGENTS
// ============================================================================
// Following the official LangChain tutorial approach
// Uses a model with native tool-calling support (qwen2.5)
// ============================================================================

logSection("🔧 Part 2B: Conversational RAG with Agents");

// ============================================================================
// LLM WITH TOOL CALLING SUPPORT
// ============================================================================
// Note: llama2 doesn't support tool calling, so we use qwen2.5
// Other options: llama3.1, mistral, qwen2.5, etc.

const llm = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "qwen2.5:3b", // Model with tool-calling support
  temperature: 0,
});

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://localhost:11434",
  model: "nomic-embed-text",
});

// ============================================================================
// DATA LOADING & INDEXING
// ============================================================================

console.log("\n📥 Loading and indexing documents");

const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  { selector: "p" }
);

const docs = await cheerioLoader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splits = await textSplitter.splitDocuments(docs);

const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
const retriever = vectorStore.asRetriever({ k: 3 });

console.log(`   ✓ Indexed ${splits.length} chunks\n`);

// ============================================================================
// CREATE RETRIEVER TOOL (Following Tutorial)
// ============================================================================

const tool = createRetrieverTool(retriever, {
  name: "retrieve_blog_posts",
  description: 
    "Search and return information about LLM agents, autonomous agents, and task decomposition from Lilian Weng's blog.",
});

const tools = [tool];

// ============================================================================
// CREATE REACT AGENT (Following Tutorial)
// ============================================================================

// @ts-expect-error - Type inference issue with createReactAgent
const agent = createReactAgent({ llm, tools });

console.log("   ✓ Agent created\n");

// ============================================================================
// INTERACTIVE SESSION
// ============================================================================

const chatHistory: BaseMessage[] = [];

async function askQuestion(question: string) {
  logQuestion(question);
  
  const startTime = Date.now();
  
  // Build messages: chat history + new question
  const messages = [...chatHistory, new HumanMessage(question)];
  
  // Stream the agent execution
  console.log("🔄 Agent execution:\n");
  
  let finalAnswer = "";
  
  for await (const step of await agent.stream(
    { messages },
    { streamMode: "values" }
  )) {
    const lastMessage = step.messages[step.messages.length - 1];
    
    // Log each step
    if (isAIMessage(lastMessage)) {
      const aiMsg = lastMessage as AIMessage;
      
      if (aiMsg.content) {
        console.log(`🤖 AI: ${aiMsg.content}`);
        finalAnswer = aiMsg.content.toString();
      } else if (aiMsg.tool_calls && aiMsg.tool_calls.length > 0) {
        const toolCall = aiMsg.tool_calls[0];
        console.log(`🔧 Tool Call: ${toolCall.name}`);
        console.log(`   → Query: "${toolCall.args.query}"`);
      }
    } else if (isToolMessage(lastMessage)) {
      console.log(`📥 Tool Result`);
      console.log(`   → Retrieved ${lastMessage.content.toString().length} chars`);
    }
    console.log();
  }
  
  const duration = (Date.now() - startTime) / 1000;
  logDivider();
  logTime(duration);
  logSeparator();
  
  // Update chat history (only Human/AI messages)
  chatHistory.push(new HumanMessage(question));
  chatHistory.push(new AIMessage(finalAnswer));
  
  return finalAnswer;
}

// ============================================================================
// RUN SESSION
// ============================================================================

logSection("🚀 STARTING AGENTIC CONVERSATIONAL RAG SESSION");

await askQuestion("What is Task Decomposition?");
await askQuestion("What are common ways of doing it?");
await askQuestion("Can you compare the different approaches and tell me which one is most commonly used?");

logSummary(chatHistory.length);
