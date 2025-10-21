import { BaseMessage, isAIMessage, AIMessage } from "@langchain/core/messages";

/**
 * Pretty prints a message with tool calls if present
 * 
 * @param message - Any LangChain BaseMessage (HumanMessage, AIMessage, ToolMessage, etc.)
 * 
 * @example
 * ```typescript
 * import { prettyPrint } from "../utils/prettyPrint";
 * 
 * // Simple message
 * prettyPrint(new HumanMessage("Hello!"));
 * // Output: [human]: Hello!
 * 
 * // AI message with tool calls
 * const aiMsg = new AIMessage({
 *   content: "",
 *   tool_calls: [{ name: "retrieve", args: { query: "test" } }]
 * });
 * prettyPrint(aiMsg);
 * // Output:
 * // [ai]: 
 * // Tools: 
 * // - retrieve({"query":"test"})
 * ```
 */
export const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message.getType()}]: ${message.content}`;
  if ((isAIMessage(message) && (message.tool_calls?.length || 0) > 0)) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

