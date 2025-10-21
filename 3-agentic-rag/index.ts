import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { AIMessage, HumanMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { logSection, logQuestion, logDivider, logTime, logSeparator } from "../utils/logger.js";

// ============================================================================
// PART 3: AGENTIC RAG WITH REACT FRAMEWORK
// ============================================================================
// ReAct pattern: Reason → Act → Observe → Learn (repeat until satisfied)
// ============================================================================

logSection("🤖 Part 3: Agentic RAG with ReAct Framework");

const llm = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "qwen2.5:3b", // Requires tool-calling support
  temperature: 0,
});

const embeddings = new OllamaEmbeddings({
  baseUrl: "http://localhost:11434",
  model: "nomic-embed-text",
});

// ============================================================================
// DATA LOADING & INDEXING
// ============================================================================

console.log("\n📥 Loading and indexing documents...");

const urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];

const docs = await Promise.all(
  urls.map((url) => new CheerioWebBaseLoader(url, { selector: "p" }).load())
);

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const docSplits = await textSplitter.splitDocuments(docs.flat());

const vectorStore = await MemoryVectorStore.fromDocuments(docSplits, embeddings);
const retriever = vectorStore.asRetriever({ k: 3 });

console.log(`✓ Indexed ${docSplits.length} chunks from ${urls.length} blog posts\n`);

// ============================================================================
// CREATE RETRIEVER TOOL
// ============================================================================

const tool = createRetrieverTool(retriever, {
  name: "retrieve_blog_posts",
  description: "Search Lilian Weng's blog posts on LLM agents, prompt engineering, and adversarial attacks.",
});

// ============================================================================
// GRAPH STATE
// ============================================================================

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  question: Annotation<string>({
    reducer: (x, y) => y ?? x ?? "",
    default: () => "",
  }),
});

// ============================================================================
// NODE 1: AGENT - Decides to retrieve or answer
// ============================================================================

async function agent(state: typeof GraphState.State) {
  const { messages } = state;
  
  console.log("🤔 Agent reasoning...");
  
  const systemPrompt = `You are an assistant for question-answering about LLM agents.

If the question is a greeting or general chat, respond directly.
If the question needs information from the blog, use the retrieve_blog_posts tool.

You have access to one tool:
- retrieve_blog_posts: Search Lilian Weng's blog posts`;

  const response = await llm.bindTools([tool]).invoke([
    { role: "system", content: systemPrompt },
    ...messages,
  ]);
  
  if (response.tool_calls && response.tool_calls.length > 0) {
    console.log(`   → Decision: Retrieve with query "${response.tool_calls[0].args.query}"`);
  } else {
    console.log(`   → Decision: Answer directly\n`);
  }
  
  return { messages: [response] };
}

// ============================================================================
// NODE 2: RETRIEVE - Execute retrieval tool
// ============================================================================

async function retrieve(state: typeof GraphState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  
  if (!lastMessage.tool_calls || lastMessage.tool_calls.length === 0) {
    return { messages: [] };
  }
  
  const toolCall = lastMessage.tool_calls[0];
  console.log(`📥 Retrieving documents...`);
  
  const docs = await retriever.invoke(toolCall.args.query);
  const content = docs.map(d => d.pageContent).join("\n\n");
  
  console.log(`   → Retrieved ${docs.length} documents\n`);
  
  return {
    messages: [
      new ToolMessage({
        content: content,
        tool_call_id: toolCall.id!,
      }),
    ],
  };
}

// ============================================================================
// NODE 3: GRADE - Check document relevance
// ============================================================================

async function gradeDocuments(state: typeof GraphState.State) {
  const { messages, question } = state;
  
  // Find the most recent tool result
  let retrievedDocs = "";
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]._getType() === "tool") {
      retrievedDocs = messages[i].content as string;
      break;
    }
  }
  
  console.log("📊 Grading document relevance...");
  
  const gradePrompt = `You are a grader assessing relevance of retrieved documents to a question.

Question: ${question}

Retrieved documents:
${retrievedDocs.substring(0, 500)}...

Are these documents relevant to the question? Answer only: relevant or not relevant`;

  const response = await llm.invoke(gradePrompt);
  const grade = response.content.toString().toLowerCase();
  
  const isRelevant = grade.includes("relevant") && !grade.includes("not relevant");
  
  console.log(`   → Grade: ${isRelevant ? "✅ Relevant" : "❌ Not relevant"}\n`);
  
  return {
    messages: [
      new AIMessage({
        content: "",
        additional_kwargs: { grade: isRelevant ? "relevant" : "not_relevant" },
      }),
    ],
  };
}

// ============================================================================
// NODE 4: REWRITE - Improve query if docs not relevant
// ============================================================================

async function rewriteQuery(state: typeof GraphState.State) {
  const { question } = state;
  
  console.log("🔄 Rewriting query...");
  
  const rewritePrompt = `Rewrite this question to improve retrieval from a blog about LLM agents:

Original question: ${question}

Provide ONLY the rewritten question, nothing else.`;

  const response = await llm.invoke(rewritePrompt);
  const rewrittenQuestion = response.content.toString().trim();
  
  console.log(`   → New query: "${rewrittenQuestion}"\n`);
  
  return {
    messages: [new HumanMessage(rewrittenQuestion)],
    question: rewrittenQuestion,
  };
}

// ============================================================================
// NODE 5: GENERATE - Create final answer
// ============================================================================

async function generate(state: typeof GraphState.State) {
  const { messages, question } = state;
  
  // Find retrieved docs
  let retrievedDocs = "";
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]._getType() === "tool") {
      retrievedDocs = messages[i].content as string;
      break;
    }
  }
  
  console.log("✍️  Generating answer...\n");
  
  const generatePrompt = `Answer the question using the context below. Be concise (2-3 sentences).

Question: ${question}

Context:
${retrievedDocs}

Answer:`;

  const response = await llm.invoke(generatePrompt);
  
  return { messages: [response] };
}

// ============================================================================
// ROUTING LOGIC
// ============================================================================

function routeAfterAgent(state: typeof GraphState.State): string {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  
  // If agent made tool call, retrieve
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return "retrieve";
  }
  // Otherwise, answer directly
  return "generate";
}

function routeAfterGrade(state: typeof GraphState.State): string {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  const grade = lastMessage.additional_kwargs?.grade;
  
  // If relevant, generate answer
  if (grade === "relevant") {
    return "generate";
  }
  // If not relevant, rewrite query
  return "rewrite";
}

// ============================================================================
// BUILD THE GRAPH
// ============================================================================

const workflow = new StateGraph(GraphState)
  .addNode("agent", agent)
  .addNode("retrieve", retrieve)
  .addNode("grade", gradeDocuments)
  .addNode("rewrite", rewriteQuery)
  .addNode("generate", generate)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", routeAfterAgent, {
    retrieve: "retrieve",
    generate: "generate",
  })
  .addEdge("retrieve", "grade")
  .addConditionalEdges("grade", routeAfterGrade, {
    generate: "generate",
    rewrite: "rewrite",
  })
  .addEdge("rewrite", "agent")
  .addEdge("generate", END);

const graph = workflow.compile();

console.log("✓ ReAct graph compiled\n");

// ============================================================================
// RUN QUESTIONS
// ============================================================================

async function askQuestion(question: string) {
  logQuestion(question);
  
  const startTime = Date.now();
  
  const result = await graph.invoke({
    messages: [new HumanMessage(question)],
    question: question,
  });
  
  // Extract final answer
  const lastMessage = result.messages[result.messages.length - 1];
  if (lastMessage.content) {
    console.log(`🤖 AI: ${lastMessage.content}\n`);
  }
  
  const duration = (Date.now() - startTime) / 1000;
  logDivider();
  logTime(duration);
  logSeparator();
}

logSection("🚀 STARTING AGENTIC RAG SESSION");

await askQuestion("What is Task Decomposition?");
await askQuestion("What are the types of agent memory?");
