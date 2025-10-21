import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { logSection, logTime, logSeparator } from "../utils/logger.js";

// ============================================================================
// PART 1: BASIC RAG (Retrieval-Augmented Generation)
// ============================================================================
// This is a basic RAG implementation without chat history.
// It demonstrates the core RAG pipeline: Load → Split → Embed → Retrieve → Generate
// ============================================================================

// ============================================================================
// CONFIGURATION & SETUP
// ============================================================================

// Connect to the Ollama instance
const llm = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "llama2",
  temperature: 0,
});

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://localhost:11434",
  model: "nomic-embed-text",
});

const vectorStore = new MemoryVectorStore(embeddings);

// ============================================================================
// DATA LOADING
// ============================================================================

console.log("\n📥 Loading and indexing documents...");
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  }
);

const docs = await cheerioLoader.load();
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const allSplits = await splitter.splitDocuments(docs);
await vectorStore.addDocuments(allSplits);
console.log(`✓ Indexed ${allSplits.length} chunks`);

// ============================================================================
// RETRIEVE AND AUGMENT (RAG PIPELINE)
// ============================================================================

const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

// Define state for application
const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

// Define application steps
const retrieve = async (state: typeof InputStateAnnotation.State) => {
  console.log(`\n🔍 Retrieving relevant documents...`);
  const retrievedDocs = await vectorStore.similaritySearch(state.question);
  console.log(`✓ Retrieved ${retrievedDocs.length} document(s)`);
  return { context: retrievedDocs };
};

const generate = async (state: typeof StateAnnotation.State) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  console.log(`\n🤖 Generating answer with llama2...`);
  console.log(`   Context: ${docsContent.length} characters from ${state.context.length} document(s)`);
  
  const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
  const response = await llm.invoke(messages);
  
  console.log(`✓ Answer generated\n`);
  return { answer: response.content };
};

// Compile application and test
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

console.log("✓ RAG graph compiled");

// ============================================================================
// EXECUTION
// ============================================================================

logSection("🚀 RUNNING BASIC RAG QUERY (Part 1)");

let inputs = { question: "What is Task Decomposition?" };
console.log(`\n❓ Question: "${inputs.question}"\n`);

const startTime = Date.now();
const result = await graph.invoke(inputs);
const duration = (Date.now() - startTime) / 1000;

logSection("📝 ANSWER:");
console.log(result.answer);
console.log();
logTime(duration);
logSeparator();

