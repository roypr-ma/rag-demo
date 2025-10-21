import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrieverTool } from "langchain/tools/retriever";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, HumanMessage, BaseMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { logSection, logQuestion, logDivider, logTime, logSeparator } from "../utils/logger.js";

// ============================================================================
// PART 3: AGENTIC RAG WITH REACT FRAMEWORK
// ============================================================================
// ReAct pattern: Reason → Act → Observe → Learn (repeat until satisfied)
// ============================================================================

logSection("🤖 Part 3: Agentic RAG with ReAct Framework");

const llm = new ChatOllama({
  baseUrl: "http://localhost:11434",
  model: "llama3.1", // Better tool-calling and instruction following
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

console.log(`   ✓ Indexed ${docSplits.length} chunks from ${urls.length} blog posts\n`);

// ============================================================================
// CREATE RETRIEVER TOOL
// ============================================================================

const tool = createRetrieverTool(retriever, {
  name: "retrieve_blog_posts",
  description: "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
});
const tools = [tool];

// Create ToolNode for retrieval
// @ts-expect-error - Type inference issue with ToolNode
const toolNode = new ToolNode(tools);

// ============================================================================
// GRAPH STATE
// ============================================================================

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

// ============================================================================
// NODE 1: GENERATE QUERY OR RESPOND
// ============================================================================

async function generateQueryOrRespond(state: typeof GraphState.State) {
  const { messages } = state;
  
  console.log("🤔 Agent reasoning");
  
  const model = llm.bindTools(tools);
  const response = await model.invoke(messages);
  
  if (response.tool_calls && response.tool_calls.length > 0) {
    console.log(`   → Decision: Retrieve with query "${response.tool_calls[0].args.query}"\n`);
  } else {
    console.log(`   → Decision: Answer directly\n`);
  }
  
  return { messages: [response] };
}

// ============================================================================
// NODE 2: GRADE DOCUMENTS
// ============================================================================

const gradePrompt = ChatPromptTemplate.fromTemplate(
  `You are a grader assessing relevance of retrieved docs to a user question.
Here are the retrieved docs:
\n ------- \n
{context}
\n ------- \n
Here is the user question: {question}
If the content of the docs are relevant to the users question, score them as relevant.
Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
Yes: The docs are relevant to the question.
No: The docs are not relevant to the question.`
);

const gradeDocumentsSchema = z.object({
  binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),
});

async function gradeDocuments(state: typeof GraphState.State) {
  const { messages } = state;
  
  console.log("📊 Grading document relevance");
  
  // @ts-expect-error - Type inference issue with structured output
  const model = llm.withStructuredOutput(gradeDocumentsSchema);
  const chain = gradePrompt.pipe(model);
  
  const score = await chain.invoke({
    question: messages[0].content,
    context: messages[messages.length - 1].content,
  });
  
  const isRelevant = score.binaryScore === "yes";
  
  console.log(`   → Grade: ${isRelevant ? "✅ Relevant" : "❌ Not relevant"}\n`);
  
  return {
    messages: [
      new AIMessage({
        content: isRelevant ? "generate" : "rewrite",
      }),
    ],
  };
}

// ============================================================================
// NODE 3: REWRITE
// ============================================================================

const rewritePrompt = ChatPromptTemplate.fromTemplate(
  `Look at the input and try to reason about the underlying semantic intent / meaning. \n
Here is the initial question:
\n ------- \n
{question}
\n ------- \n
Formulate an improved question:`
);

async function rewrite(state: typeof GraphState.State) {
  const { messages } = state;
  const question = messages[0].content;
  
  console.log("🔄 Rewriting query");
  
  const response = await rewritePrompt.pipe(llm).invoke({ question });
  
  console.log(`   → New query: "${response.content}"\n`);
  
  return {
    messages: [response],
  };
}

// ============================================================================
// NODE 4: GENERATE
// ============================================================================

async function generate(state: typeof GraphState.State) {
  const { messages } = state;
  const question = messages[0].content;
  const context = messages[messages.length - 1].content;
  
  console.log("✍️  Generating answer\n");
  
  const prompt = ChatPromptTemplate.fromTemplate(
    `You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}`
  );
  
  const ragChain = prompt.pipe(llm);
  
  const response = await ragChain.invoke({
    context,
    question,
  });
  
  return {
    messages: [response],
  };
}

// ============================================================================
// ROUTING LOGIC
// ============================================================================

// Helper function to determine if we should retrieve
function shouldRetrieve(state: typeof GraphState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  
  if (lastMessage.tool_calls && lastMessage.tool_calls.length) {
    return "retrieve";
  }
  return END;
}

function checkRelevance(state: typeof GraphState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  
  // gradeDocuments returns either "generate" or "rewrite"
  return lastMessage.content === "generate" ? "generate" : "rewrite";
}

// ============================================================================
// BUILD THE GRAPH
// ============================================================================

const workflow = new StateGraph(GraphState)
  .addNode("generateQueryOrRespond", generateQueryOrRespond)
  .addNode("retrieve", toolNode)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)
  // Add edges
  .addEdge(START, "generateQueryOrRespond")
  // Decide whether to retrieve
  .addConditionalEdges("generateQueryOrRespond", shouldRetrieve)
  .addEdge("retrieve", "gradeDocuments")
  // Edges taken after grading documents
  .addConditionalEdges("gradeDocuments", checkRelevance)
  .addEdge("generate", END)
  .addEdge("rewrite", "generateQueryOrRespond");

const graph = workflow.compile();

console.log("   ✓ ReAct graph compiled\n");

// ============================================================================
// RUN QUESTIONS
// ============================================================================

async function askQuestion(question: string) {
  logQuestion(question);
  
  const startTime = Date.now();
  
  const result = await graph.invoke({
    messages: [new HumanMessage(question)],
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
