import axios from "axios";

const tbServerUrl = "http://localhost:8080";

export interface ChatMessage {
  role: "user" | "assistant" | "tool" | "system";
  content: string;
  tool_calls?: any[];
}

export interface ChatResponse {
  reply: string;
  error?: string;
  tool_log: ToolLogEntry[];
}

export interface ToolLogEntry {
  tool: string;
  args: string;
  result: string;
  elapsed_ms: number;
}

export async function sendMessage(
  messages: ChatMessage[],
  model: string
): Promise<ChatResponse> {
  const response = await axios.post(`${tbServerUrl}/api/chat`, {
    messages,
    model,
  });
  return response.data;
}

export async function getModels(): Promise<string[]> {
  try {
    const response = await axios.get(`${tbServerUrl}/api/chat/models`);
    return response.data.models || [];
  } catch {
    return [];
  }
}
