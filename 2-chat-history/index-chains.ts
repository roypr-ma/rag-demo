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
import { logSection, logQuestion, logDivider, logTime, logSeparator, logSummary } from "../utils/logger.js";

// ============================================================================
// PART 2A: RAG WITH CHAT HISTORY (Chains Approach)
// ============================================================================
// This implementation adds conversational memory to RAG using chains.
// CHAINS approach: Predictable flow with exactly ONE retrieval per question.
// It uses:
// - createHistoryAwareRetriever: Reformulates questions based on chat history
// - createRetrievalChain: Orchestrates retrieval and generation
// - Chat history management: Maintains context across multiple turns
// 
// See index-agents.ts for the AGENTS approach (multiple retrievals per question)
// ============================================================================

// ============================================================================
// CONFIGURATION & SETUP
// ============================================================================

logSection("🔧 Part 2A: RAG with Chat History (Chains Approach)");

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
const retriever = vectorStore.asRetriever();
console.log(`   ✓ Indexed ${splits.length} chunks\n`);

// ============================================================================
// HISTORY-AWARE RETRIEVER
// ============================================================================
// This retriever reformulates the user's question based on chat history
// before performing the search, ensuring context is preserved.

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

// ============================================================================
// QUESTION ANSWERING CHAIN
// ============================================================================
// This chain answers questions using the retrieved context and chat history.

console.log("💬 Setting up Q&A chain");

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

console.log("   ✓ Ready\n");

// ============================================================================
// STATEFUL CHAT HISTORY MANAGEMENT
// ============================================================================

const chatHistory: BaseMessage[] = [];

async function askQuestion(question: string) {
  logQuestion(question);

  const startTime = Date.now();
  const result = await ragChain.invoke({
    input: question,
    chat_history: chatHistory,
  });
  const duration = (Date.now() - startTime) / 1000;

  const answer = result.answer;
  console.log(`🤖 AI: ${answer}`);
  logDivider();
  logTime(duration);
  logSeparator();

  // Update chat history
  chatHistory.push(new HumanMessage(question));
  chatHistory.push(new AIMessage(answer));

  return answer;
}

// ============================================================================
// EXECUTION: CONVERSATIONAL INTERACTION
// ============================================================================

logSection("🚀 STARTING CONVERSATIONAL RAG SESSION");

// First question
await askQuestion("What is Task Decomposition?");

// Follow-up question (references previous context)
await askQuestion("What are common ways of doing it?");

// Another follow-up (continues the conversation)
await askQuestion("Can you give me specific examples?");

logSummary(chatHistory.length);

