import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { AIMessage, HumanMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";
import { prettyPrint } from "../utils/prettyPrint";

// ============================================================================
// PART 3: AGENTIC RAG WITH LANGGRAPH
// ============================================================================
// This implementation creates an intelligent RAG agent that can:
// 1. Decide whether to retrieve documents or respond directly
// 2. Grade retrieved documents for relevance
// 3. Rewrite queries if documents aren't relevant
// 4. Generate answers using only relevant context
// ============================================================================

// ============================================================================
// CONFIGURATION & SETUP
// ============================================================================

console.log("\n" + "=".repeat(70));
console.log("🤖 Part 3: Agentic RAG with LangGraph");
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
// STEP 1: PREPROCESS DOCUMENTS
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

const docsList = docs.flat();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const docSplits = await textSplitter.splitDocuments(docsList);
console.log(`✓ Indexed ${docSplits.length} chunks from ${urls.length} blog posts\n`);

// ============================================================================
// STEP 2: CREATE A RETRIEVER TOOL
// ============================================================================

const vectorStore = await MemoryVectorStore.fromDocuments(
  docSplits,
  embeddings
);
const retriever = vectorStore.asRetriever({ k: 3 });

// Define tool information for the LLM (prompt-based approach for llama2 compatibility)
const toolDefinition = {
  name: "retrieve_blog_posts",
  description:
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
};

// ============================================================================
// GRAPH STATE DEFINITION
// ============================================================================

const GraphState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

// ============================================================================
// STEP 3: GENERATE QUERY OR RESPOND
// ============================================================================

async function generateQueryOrRespond(state: typeof GraphState.State) {
  const { messages } = state;
  
  // Use prompt engineering for tool calling (llama2 compatible)
  const systemPrompt = `You are a helpful assistant with access to a retrieval tool.

Tool available: retrieve_blog_posts
Description: ${toolDefinition.description}

If the user asks a question that requires information from Lilian Weng's blog posts about LLM agents, prompt engineering, or adversarial attacks, respond with:
TOOL_CALL: retrieve_blog_posts
QUERY: <your search query>

Otherwise, respond normally to the user.`;

  const promptedMessages = [
    new HumanMessage(systemPrompt),
    ...messages
  ];
  
  const response = await llm.invoke(promptedMessages);
  const content = response.content.toString();
  
  // Check if model wants to use the tool
  if (content.includes("TOOL_CALL: retrieve_blog_posts")) {
    const queryMatch = content.match(/QUERY: (.+)/);
    const query = queryMatch ? queryMatch[1].trim() : messages[messages.length - 1].content.toString();
    
    // Create AI message with tool call metadata
    const aiMessage = new AIMessage({
      content: "",
      additional_kwargs: {
        tool_call: true,
        tool_query: query,
      },
    });
    
    return { messages: [aiMessage] };
  } else {
    return { messages: [response] };
  }
}

// ============================================================================
// STEP 4: GRADE DOCUMENTS
// ============================================================================

async function gradeDocuments(state: typeof GraphState.State) {
  const { messages } = state;
  const question = messages[0].content;
  const toolMessage = messages[messages.length - 1];
  const docs = toolMessage.content;

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are a grader assessing relevance of retrieved documents to a user question.
    
Here are the retrieved documents:
---
{context}
---

Here is the user question: {question}

If the documents contain information relevant to the user's question, respond with just the word "yes".
If the documents are not relevant, respond with just the word "no".

Your response (yes/no):`
  );

  const chain = prompt.pipe(llm);
  const score = await chain.invoke({
    question: question,
    context: docs,
  });

  const gradeResult = score.content.toString().toLowerCase().trim();
  
  if (gradeResult.includes("yes")) {
    return {
      messages: [new AIMessage("generate")],
    };
  } else {
    return {
      messages: [new AIMessage("rewrite")],
    };
  }
}

// ============================================================================
// STEP 5: REWRITE QUESTION
// ============================================================================

async function rewrite(state: typeof GraphState.State) {
  const { messages } = state;
  const question = messages[0].content;

  const rewritePrompt = ChatPromptTemplate.fromTemplate(
    `Look at the input and try to reason about the underlying semantic intent/meaning.

Here is the initial question:
---
{question}
---

Formulate an improved question that is more specific and likely to retrieve relevant information:`
  );

  const response = await rewritePrompt.pipe(llm).invoke({ question });
  
  const rewrittenQuestion = response.content.toString();
  
  return {
    messages: [new HumanMessage(rewrittenQuestion)],
  };
}

// ============================================================================
// STEP 6: GENERATE ANSWER
// ============================================================================

async function generate(state: typeof GraphState.State) {
  const { messages } = state;
  const question = messages[0].content;
  
  // Find the last tool message (contains retrieved docs)
  let context = "";
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]._getType() === "tool") {
      context = messages[i].content.toString();
      break;
    }
  }

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:`
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
// STEP 7: ASSEMBLE THE GRAPH
// ============================================================================

// Node to execute retrieval manually (llama2 compatible approach)
async function retrieve(state: typeof GraphState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  
  if (lastMessage instanceof AIMessage && lastMessage.additional_kwargs?.tool_call) {
    const query = lastMessage.additional_kwargs.tool_query as string;
    
    const docs = await retriever.invoke(query);
    const context = docs.map(doc => doc.pageContent).join("\n\n");
    
    const toolMessage = new ToolMessage({
      content: context,
      tool_call_id: "retrieve_blog_posts",
    });
    
    return { messages: [toolMessage] };
  }
  
  return { messages: [] };
}

// Helper function to determine if we should retrieve
function shouldRetrieve(state: typeof GraphState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  if (lastMessage instanceof AIMessage && lastMessage.additional_kwargs?.tool_call) {
    return "retrieve";
  }
  return "end";
}

// Helper function to check grading result
function checkRelevance(state: typeof GraphState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  
  if (lastMessage.content === "generate") {
    return "generate";
  } else {
    return "rewrite";
  }
}

// Define the graph
const graph = new StateGraph(GraphState)
  .addNode("generateQueryOrRespond", generateQueryOrRespond)
  .addNode("retrieve", retrieve)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)
  // Add edges
  .addEdge(START, "generateQueryOrRespond")
  // Decide whether to retrieve
  .addConditionalEdges("generateQueryOrRespond", shouldRetrieve, {
    retrieve: "retrieve",
    end: END,
  })
  .addEdge("retrieve", "gradeDocuments")
  // Grade and route
  .addConditionalEdges("gradeDocuments", checkRelevance, {
    generate: "generate",
    rewrite: "rewrite",
  })
  .addEdge("generate", END)
  .addEdge("rewrite", "generateQueryOrRespond")
  .compile();

console.log("✓ Graph compiled\n");

// ============================================================================
// STEP 8: RUN THE AGENTIC RAG
// ============================================================================

console.log("\n" + "=".repeat(70));
console.log("🚀 RUNNING AGENTIC RAG QUERIES");
console.log("=".repeat(70));

// Test 1: Question requiring retrieval
console.log("\n" + "=".repeat(70));
console.log("TEST 1: Complex question requiring retrieval");
console.log("=".repeat(70));

const question1 = "What are the types of memory in LLM agents?";
console.log(`\n❓ "${question1}"\n`);

const startTime1 = Date.now();
let result1;

console.log("🔄 ReAct Cycle:\n");
for await (const output of await graph.stream({
  messages: [new HumanMessage(question1)],
})) {
  const nodeOutput = Object.values(output)[0];
  // Show the last message added by each node
  const lastMsg = nodeOutput.messages[nodeOutput.messages.length - 1];
  if (lastMsg) {
    prettyPrint(lastMsg);
  }
  result1 = nodeOutput;
}

const duration1 = ((Date.now() - startTime1) / 1000).toFixed(2);
const finalAnswer1 = result1!.messages[result1!.messages.length - 1];

console.log(`\n💬 Final Answer:\n${finalAnswer1.content}`);
console.log(`\n⏱️  ${duration1}s`);
console.log("=".repeat(70));

// Test 2: Simple greeting (should not retrieve)
console.log("\n" + "=".repeat(70));
console.log("TEST 2: Simple greeting (no retrieval needed)");
console.log("=".repeat(70));

const question2 = "Hello! How are you?";
console.log(`\n❓ "${question2}"\n`);

const startTime2 = Date.now();
let result2;

console.log("🔄 ReAct Cycle:\n");
for await (const output of await graph.stream({
  messages: [new HumanMessage(question2)],
})) {
  const nodeOutput = Object.values(output)[0];
  // Show the last message added by each node
  const lastMsg = nodeOutput.messages[nodeOutput.messages.length - 1];
  if (lastMsg) {
    prettyPrint(lastMsg);
  }
  result2 = nodeOutput;
}

const duration2 = ((Date.now() - startTime2) / 1000).toFixed(2);
const finalAnswer2 = result2!.messages[result2!.messages.length - 1];

console.log(`\n💬 Final Answer:\n${finalAnswer2.content}`);
console.log(`\n⏱️  ${duration2}s`);
console.log("=".repeat(70));

// Test 3: Question about prompt engineering
console.log("\n" + "=".repeat(70));
console.log("TEST 3: Specific question about prompt engineering");
console.log("=".repeat(70));

const question3 = "What does Lilian Weng say about chain of thought prompting?";
console.log(`\n❓ "${question3}"\n`);

const startTime3 = Date.now();
let result3;

console.log("🔄 ReAct Cycle:\n");
for await (const output of await graph.stream({
  messages: [new HumanMessage(question3)],
})) {
  const nodeOutput = Object.values(output)[0];
  // Show the last message added by each node
  const lastMsg = nodeOutput.messages[nodeOutput.messages.length - 1];
  if (lastMsg) {
    prettyPrint(lastMsg);
  }
  result3 = nodeOutput;
}

const duration3 = ((Date.now() - startTime3) / 1000).toFixed(2);
const finalAnswer3 = result3!.messages[result3!.messages.length - 1];

console.log(`\n💬 Final Answer:\n${finalAnswer3.content}`);
console.log(`\n⏱️  ${duration3}s`);
console.log("=".repeat(70));

