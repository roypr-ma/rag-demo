import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { AIMessage, BaseMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { StateGraph, START, END, Annotation } from "@langchain/langgraph";
import { prettyPrint } from "../utils/prettyPrint";

// ============================================================================
// PART 2B: CONVERSATIONAL RAG WITH AGENTS
// ============================================================================
// This implementation uses an agent that can make multiple retrieval calls
// per question, deciding how many times to retrieve based on the complexity.
// ============================================================================

console.log("\n" + "=".repeat(70));
console.log("🔧 Part 2B: Conversational RAG with Agents");
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

console.log("\n📥 Loading and indexing documents...");

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

console.log(`✓ Indexed ${splits.length} chunks\n`);

// ============================================================================
// DEFINE RETRIEVER TOOL
// ============================================================================

const toolDefinition = {
  name: "retrieve",
  description:
    "Search for information about LLM agents, task decomposition, and autonomous agents. Use this when you need specific information from the knowledge base.",
};

// ============================================================================
// GRAPH STATE DEFINITION
// ============================================================================

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

// ============================================================================
// AGENT NODE - DECIDES TO RETRIEVE OR RESPOND
// ============================================================================

async function agent(state: typeof AgentState.State) {
  const { messages } = state;
  
  // Create system prompt with chat history
  const systemPrompt = `You are a helpful assistant with access to a retrieval tool for information about LLM agents.

Tool available: retrieve
Description: ${toolDefinition.description}

If you need to retrieve information, respond with:
TOOL_CALL: retrieve
QUERY: <your search query>

You can call the tool multiple times if needed to gather all necessary information.
When you have enough information, provide a complete answer to the user.

Chat history is provided for context.`;

  const prompt = ChatPromptTemplate.fromMessages([
    ["system", systemPrompt],
    new MessagesPlaceholder("messages"),
  ]);

  const promptedMessages = await prompt.invoke({ messages });
  const response = await llm.invoke(promptedMessages.toChatMessages());
  const content = response.content.toString();
  
  // Check if model wants to use the tool
  if (content.includes("TOOL_CALL: retrieve")) {
    const queryMatch = content.match(/QUERY: (.+)/);
    const query = queryMatch ? queryMatch[1].trim() : "";
    
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
// RETRIEVE NODE - EXECUTES RETRIEVAL
// ============================================================================

async function executeRetrieval(state: typeof AgentState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];
  
  if (lastMessage instanceof AIMessage && lastMessage.additional_kwargs?.tool_call) {
    const query = lastMessage.additional_kwargs.tool_query as string;
    
    const docs = await retriever.invoke(query);
    const context = docs.map(doc => doc.pageContent).join("\n\n");
    
    const toolMessage = new ToolMessage({
      content: `Retrieved information:\n${context}`,
      tool_call_id: "retrieve",
    });
    
    return { messages: [toolMessage] };
  }
  
  return { messages: [] };
}

// ============================================================================
// ROUTING FUNCTIONS
// ============================================================================

function shouldContinue(state: typeof AgentState.State) {
  const { messages } = state;
  const lastMessage = messages[messages.length - 1];

  // If agent wants to use tool, go to retrieve
  if (lastMessage instanceof AIMessage && lastMessage.additional_kwargs?.tool_call) {
    return "retrieve";
  }
  // Otherwise end (agent has final answer)
  return "end";
}

// ============================================================================
// BUILD THE GRAPH
// ============================================================================

const graph = new StateGraph(AgentState)
  .addNode("agent", agent)
  .addNode("retrieve", executeRetrieval)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, {
    retrieve: "retrieve",
    end: END,
  })
  .addEdge("retrieve", "agent") // After retrieval, go back to agent
  .compile();

console.log("✓ Agent graph compiled\n");

// ============================================================================
// CHAT HISTORY MANAGEMENT
// ============================================================================

const chatHistory: BaseMessage[] = [];

async function askQuestion(question: string) {
  console.log("\n" + "=".repeat(70));
  console.log(`👤 Human: ${question}`);
  console.log("-".repeat(70));

  const startTime = Date.now();
  
  // Add user question to messages
  const userMessage = new HumanMessage(question);
  const initialMessages = [...chatHistory, userMessage];
  
  // Stream the agent's execution
  let finalState: any = { messages: [] };
  console.log("\n🔄 Agent execution:\n");
  
  for await (const state of await graph.stream({ messages: initialMessages })) {
    const stateValue = Object.values(state)[0] as any;
    // Show each new message
    const lastMsg = stateValue.messages[stateValue.messages.length - 1];
    if (lastMsg) {
      prettyPrint(lastMsg);
    }
    finalState = stateValue;
  }
  
  const duration = ((Date.now() - startTime) / 1000).toFixed(2);
  
  // Get the final answer (last message that's not a tool message)
  const messages = finalState.messages;
  let answer = "";
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i] instanceof AIMessage && messages[i].content) {
      answer = messages[i].content.toString();
      break;
    }
  }
  
  console.log(`🤖 AI: ${answer}`);
  console.log("-".repeat(70));
  console.log(`⏱️  Time: ${duration}s`);
  console.log("=".repeat(70));

  // Update chat history with user message and AI response
  chatHistory.push(new HumanMessage(question));
  chatHistory.push(new AIMessage(answer));

  return answer;
}

// ============================================================================
// EXECUTION: CONVERSATIONAL INTERACTION WITH AGENT
// ============================================================================

console.log("\n" + "=".repeat(70));
console.log("🚀 STARTING AGENTIC CONVERSATIONAL RAG SESSION");
console.log("=".repeat(70));

// Simple question
await askQuestion("What is Task Decomposition?");

// Follow-up using history
await askQuestion("What are common ways of doing it?");

// Complex question that may require multiple retrievals
await askQuestion("Can you compare the different approaches and tell me which one is most commonly used?");

console.log("\n" + "=".repeat(70));
console.log("📊 SUMMARY");
console.log("=".repeat(70));
console.log(`💾 Total messages in history: ${chatHistory.length}`);
console.log("=".repeat(70));

