import { Module } from "vuex";
import { State } from "../index";
import * as chatService from "@/services/chatService";

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  toolLog?: chatService.ToolLogEntry[];
}

export interface ChatState {
  messages: ChatMessage[];
  isLoading: boolean;
  currentModel: string;
  models: string[];
}

const chatModule: Module<ChatState, State> = {
  namespaced: true,
  state: () => ({
    messages: [],
    isLoading: false,
    currentModel: "qwen3-coder:30b",
    models: [],
  }),
  mutations: {
    addMessage(state, message: ChatMessage) {
      state.messages.push(message);
    },
    setLoading(state, loading: boolean) {
      state.isLoading = loading;
    },
    setModel(state, model: string) {
      state.currentModel = model;
    },
    setModels(state, models: string[]) {
      state.models = models;
    },
    clearMessages(state) {
      state.messages = [];
    },
  },
  actions: {
    async loadModels({ commit }) {
      const models = await chatService.getModels();
      commit("setModels", models);
      if (models.length > 0 && !models.includes("qwen3-coder:30b")) {
        commit("setModel", models[0]);
      }
    },
    async sendMessage({ commit, state }, content: string) {
      // Add user message
      commit("addMessage", { role: "user", content });
      commit("setLoading", true);

      try {
        // Build message history for API (without toolLog field)
        const apiMessages: chatService.ChatMessage[] = state.messages.map(
          (m) => ({
            role: m.role,
            content: m.content,
          })
        );

        const response = await chatService.sendMessage(
          apiMessages,
          state.currentModel
        );

        // Add assistant response
        commit("addMessage", {
          role: "assistant",
          content: response.reply || response.error || "No response",
          toolLog: response.tool_log,
        });
      } catch (error: any) {
        commit("addMessage", {
          role: "assistant",
          content: `Error: ${error.message || "Failed to connect to chat server"}`,
        });
      } finally {
        commit("setLoading", false);
      }
    },
  },
};

export default chatModule;
