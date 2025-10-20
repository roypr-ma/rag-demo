import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { AIMessage, HumanMessage, BaseMessage, ToolMessage } from "@langchain/core/messages";

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
console.log("🤖 Initializing Part 3: Agentic RAG with LangGraph");
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

console.log("\n📥 Step 1: Loading and preprocessing documents...");

const urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];

console.log(`   Loading ${urls.length} blog posts...`);
const docs = await Promise.all(
  urls.map((url) => new CheerioWebBaseLoader(url, { selector: "p" }).load())
);

const docsList = docs.flat();
console.log(`✓ Loaded ${docsList.length} documents`);

console.log("\n✂️  Splitting documents into chunks...");
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const docSplits = await textSplitter.splitDocuments(docsList);
console.log(`✓ Created ${docSplits.length} chunks`);

// ============================================================================
// STEP 2: CREATE A RETRIEVER TOOL
// ============================================================================

console.log("\n🔧 Step 2: Creating retriever tool...");

const vectorStore = await MemoryVectorStore.fromDocuments(
  docSplits,
  embeddings
);
const retriever = vectorStore.asRetriever({ k: 3 });

// Define tool information for the LLM
const toolDefinition = {
  name: "retrieve_blog_posts",
  description:
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs. Input should be a search query string.",
  parameters: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "The search query to find relevant blog content",
      },
    },
    required: ["query"],
  },
};

console.log(`✓ Retriever tool defined: ${toolDefinition.name}`);

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
  console.log("\n🤔 Agent: Deciding whether to retrieve or respond...");
  
  const { messages } = state;
  
  // Create a system message that instructs the model about tool availability
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
    
    console.log(`✓ Decision: Retrieve documents with query: "${query}"`);
    
    // Create an AI message with tool call information
    const aiMessage = new AIMessage({
      content: "",
      additional_kwargs: {
        tool_call: true,
        tool_name: "retrieve_blog_posts",
        tool_query: query,
      },
    });
    
    return { messages: [aiMessage] };
  } else {
    console.log("✓ Decision: Respond directly (no retrieval needed)");
    return { messages: [response] };
  }
}

// ============================================================================
// STEP 4: GRADE DOCUMENTS
// ============================================================================

async function gradeDocuments(state: typeof GraphState.State) {
  console.log("\n📊 Grading document relevance...");
  
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
  
  console.log(`   Relevance score: ${gradeResult}`);
  
  if (gradeResult.includes("yes")) {
    console.log("✓ Documents are relevant - proceeding to generate answer");
    return {
      messages: [new AIMessage("generate")],
    };
  } else {
    console.log("✗ Documents not relevant - will rewrite query");
    return {
      messages: [new AIMessage("rewrite")],
    };
  }
}

// ============================================================================
// STEP 5: REWRITE QUESTION
// ============================================================================

async function rewrite(state: typeof GraphState.State) {
  console.log("\n✍️  Rewriting question for better retrieval...");
  
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
  console.log(`   Original: ${question}`);
  console.log(`   Rewritten: ${rewrittenQuestion}`);
  console.log("✓ Question rewritten");
  
  return {
    messages: [new HumanMessage(rewrittenQuestion)],
  };
}

// ============================================================================
// STEP 6: GENERATE ANSWER
// ============================================================================

async function generate(state: typeof GraphState.State) {
  console.log("\n💬 Generating final answer...");
  
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

  console.log("✓ Answer generated");
  
  return {
    messages: [response],
  };
}

// ============================================================================
// STEP 7: ASSEMBLE THE GRAPH
// ============================================================================

// Node to execute the retriever tool
async function retrieve(state: typeof GraphState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  
  if (lastMessage instanceof AIMessage && lastMessage.additional_kwargs?.tool_call) {
    const query = lastMessage.additional_kwargs.tool_query as string;
    console.log(`\n🔍 Retrieving documents for: "${query}"`);
    
    const docs = await retriever.invoke(query);
    const context = docs.map(doc => doc.pageContent).join("\n\n");
    
    console.log(`✓ Retrieved ${docs.length} documents`);
    
    const toolMessage = new ToolMessage({
      content: context,
      tool_call_id: "retrieve_blog_posts",
    });
    
    return { messages: [toolMessage] };
  }
  
  return { messages: [] };
}

console.log("\n🔨 Step 7: Assembling the agentic RAG graph...");

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

console.log("✓ Agentic RAG graph compiled");
console.log("\n📋 Graph structure:");
console.log("   START → generateQueryOrRespond → [retrieve OR end]");
console.log("   retrieve → gradeDocuments → [generate OR rewrite]");
console.log("   generate → END");
console.log("   rewrite → generateQueryOrRespond (loop)");

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
console.log(`\n❓ Question: "${question1}"`);

const startTime1 = Date.now();
let step = 0;

for await (const output of await graph.stream({
  messages: [new HumanMessage(question1)],
})) {
  step++;
  const nodeName = Object.keys(output)[0];
  console.log(`\n[Step ${step}] Node: ${nodeName}`);
}

const result1 = await graph.invoke({
  messages: [new HumanMessage(question1)],
});

const duration1 = ((Date.now() - startTime1) / 1000).toFixed(2);
const finalAnswer1 = result1.messages[result1.messages.length - 1];

console.log("\n" + "=".repeat(70));
console.log("📝 FINAL ANSWER:");
console.log("=".repeat(70));
console.log(finalAnswer1.content);
console.log("\n" + "=".repeat(70));
console.log(`⏱️  Time: ${duration1}s`);
console.log("=".repeat(70));

// Test 2: Simple greeting (should not retrieve)
console.log("\n" + "=".repeat(70));
console.log("TEST 2: Simple greeting (no retrieval needed)");
console.log("=".repeat(70));

const question2 = "Hello! How are you?";
console.log(`\n❓ Question: "${question2}"`);

const startTime2 = Date.now();
const result2 = await graph.invoke({
  messages: [new HumanMessage(question2)],
});

const duration2 = ((Date.now() - startTime2) / 1000).toFixed(2);
const finalAnswer2 = result2.messages[result2.messages.length - 1];

console.log("\n" + "=".repeat(70));
console.log("📝 FINAL ANSWER:");
console.log("=".repeat(70));
console.log(finalAnswer2.content);
console.log("\n" + "=".repeat(70));
console.log(`⏱️  Time: ${duration2}s`);
console.log("=".repeat(70));

// Test 3: Question about prompt engineering
console.log("\n" + "=".repeat(70));
console.log("TEST 3: Specific question about prompt engineering");
console.log("=".repeat(70));

const question3 = "What does Lilian Weng say about chain of thought prompting?";
console.log(`\n❓ Question: "${question3}"`);

const startTime3 = Date.now();
const result3 = await graph.invoke({
  messages: [new HumanMessage(question3)],
});

const duration3 = ((Date.now() - startTime3) / 1000).toFixed(2);
const finalAnswer3 = result3.messages[result3.messages.length - 1];

console.log("\n" + "=".repeat(70));
console.log("📝 FINAL ANSWER:");
console.log("=".repeat(70));
console.log(finalAnswer3.content);
console.log("\n" + "=".repeat(70));
console.log(`⏱️  Time: ${duration3}s`);
console.log("=".repeat(70));

console.log("\n" + "=".repeat(70));
console.log("✅ AGENTIC RAG DEMO COMPLETE");
console.log("=".repeat(70));
console.log("\n🎯 Key Features Demonstrated:");
console.log("   ✓ Intelligent decision-making (retrieve vs respond)");
console.log("   ✓ Document relevance grading");
console.log("   ✓ Query rewriting for better results");
console.log("   ✓ Multi-document retrieval");
console.log("   ✓ Conditional graph execution flow");
console.log("=".repeat(70));

