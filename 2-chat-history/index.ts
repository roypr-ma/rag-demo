import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// ============================================================================
// PART 2: RAG WITH CHAT HISTORY (Chains Approach)
// ============================================================================
// This implementation adds conversational memory to RAG using chains.
// It uses:
// - createHistoryAwareRetriever: Reformulates questions based on chat history
// - createRetrievalChain: Orchestrates retrieval and generation
// - Chat history management: Maintains context across multiple turns
// ============================================================================

// ============================================================================
// CONFIGURATION & SETUP
// ============================================================================

console.log("\n" + "=".repeat(70));
console.log("🔧 Initializing Part 2: RAG with Chat History (Chains)");
console.log("=".repeat(70));

const llm = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "llama2",
  temperature: 0,
});

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://localhost:11434",
  model: "nomic-embed-text",
});

// ============================================================================
// DATA LOADING & INDEXING
// ============================================================================

console.log("\n📥 Loading web content...");
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  { selector: "p" }
);

const docs = await cheerioLoader.load();
console.log(`✓ Loaded ${docs.length} document(s)`);

console.log("\n✂️  Splitting documents into chunks...");
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});
const splits = await textSplitter.splitDocuments(docs);
console.log(`✓ Created ${splits.length} chunks`);

console.log("\n📊 Creating vector store...");
const vectorStore = await MemoryVectorStore.fromDocuments(splits, embeddings);
const retriever = vectorStore.asRetriever();
console.log("✓ Vector store ready");

// ============================================================================
// HISTORY-AWARE RETRIEVER
// ============================================================================
// This retriever reformulates the user's question based on chat history
// before performing the search, ensuring context is preserved.

console.log("\n🧠 Setting up history-aware retriever...");

const contextualizeQSystemPrompt = `Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.`;

const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
  ["system", contextualizeQSystemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
]);

const historyAwareRetriever = await createHistoryAwareRetriever({
  llm,
  retriever,
  rephrasePrompt: contextualizeQPrompt,
});

console.log("✓ History-aware retriever created");

// ============================================================================
// QUESTION ANSWERING CHAIN
// ============================================================================
// This chain answers questions using the retrieved context and chat history.

console.log("\n💬 Setting up Q&A chain...");

const qaSystemPrompt = `You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}`;

const qaPrompt = ChatPromptTemplate.fromMessages([
  ["system", qaSystemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
]);

const questionAnswerChain = await createStuffDocumentsChain({
  llm,
  prompt: qaPrompt,
});

const ragChain = await createRetrievalChain({
  retriever: historyAwareRetriever,
  combineDocsChain: questionAnswerChain,
});

console.log("✓ RAG chain with history ready");

// ============================================================================
// STATEFUL CHAT HISTORY MANAGEMENT
// ============================================================================

const chatHistory: BaseMessage[] = [];

async function askQuestion(question: string) {
  console.log("\n" + "=".repeat(70));
  console.log(`👤 Human: ${question}`);
  console.log("-".repeat(70));

  const startTime = Date.now();
  const result = await ragChain.invoke({
    input: question,
    chat_history: chatHistory,
  });
  const duration = ((Date.now() - startTime) / 1000).toFixed(2);

  const answer = result.answer;
  console.log(`🤖 AI: ${answer}`);
  console.log("-".repeat(70));
  console.log(`⏱️  Time: ${duration}s`);
  console.log("=".repeat(70));

  // Update chat history
  chatHistory.push(new HumanMessage(question));
  chatHistory.push(new AIMessage(answer));

  return answer;
}

// ============================================================================
// EXECUTION: CONVERSATIONAL INTERACTION
// ============================================================================

console.log("\n" + "=".repeat(70));
console.log("🚀 STARTING CONVERSATIONAL RAG SESSION");
console.log("=".repeat(70));

// First question
await askQuestion("What is Task Decomposition?");

// Follow-up question (references previous context)
await askQuestion("What are common ways of doing it?");

// Another follow-up (continues the conversation)
await askQuestion("Can you give me specific examples?");

console.log("\n" + "=".repeat(70));
console.log("📊 SUMMARY");
console.log("=".repeat(70));
console.log(`💾 Total messages in history: ${chatHistory.length}`);
console.log("=".repeat(70));

