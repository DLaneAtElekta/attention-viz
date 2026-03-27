<template>
  <div class="chat-panel">
    <div class="chat-header">
      <span class="chat-title">Chat</span>
      <select v-model="selectedModel" class="model-select" @change="onModelChange">
        <option v-for="m in models" :key="m" :value="m">{{ m }}</option>
      </select>
    </div>
    <div class="chat-messages" ref="messageList">
      <div v-if="messages.length === 0" class="chat-empty">
        Ask about tensorBASIC programs...
      </div>
      <ChatMessage
        v-for="(msg, i) in messages"
        :key="i"
        :message="msg"
      />
      <div v-if="isLoading" class="chat-loading">Thinking...</div>
    </div>
    <div class="chat-input-area">
      <textarea
        v-model="inputText"
        class="chat-input"
        placeholder="Type a message..."
        @keydown.enter.exact.prevent="send"
        rows="2"
      ></textarea>
      <button class="chat-send" @click="send" :disabled="isLoading || !inputText.trim()">
        Send
      </button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted, watch, nextTick } from "vue";
import { useStore } from "@/store/index";
import ChatMessage from "./ChatMessage.vue";

export default defineComponent({
  name: "ChatPanel",
  components: { ChatMessage },
  setup() {
    const store = useStore();
    const inputText = ref("");
    const messageList = ref<HTMLElement | null>(null);

    const messages = computed(() => (store.state as any).chat.messages);
    const isLoading = computed(() => (store.state as any).chat.isLoading);
    const models = computed(() => (store.state as any).chat.models);
    const selectedModel = ref("");

    onMounted(async () => {
      await store.dispatch("chat/loadModels");
      selectedModel.value = (store.state as any).chat.currentModel;
    });

    const onModelChange = () => {
      store.commit("chat/setModel", selectedModel.value);
    };

    const send = async () => {
      const text = inputText.value.trim();
      if (!text || isLoading.value) return;
      inputText.value = "";
      await store.dispatch("chat/sendMessage", text);

      // Check if any tb_edit tool was called — refresh viz
      const lastMsg = messages.value[messages.value.length - 1];
      if (lastMsg?.toolLog?.some((t: any) => t.tool === "tb_edit" || t.tool === "tb_create")) {
        store.dispatch("tensorbasic/refreshFile");
      }
    };

    // Auto-scroll on new messages
    watch(messages, async () => {
      await nextTick();
      if (messageList.value) {
        messageList.value.scrollTop = messageList.value.scrollHeight;
      }
    }, { deep: true });

    return {
      inputText,
      messageList,
      messages,
      isLoading,
      models,
      selectedModel,
      onModelChange,
      send,
    };
  },
});
</script>

<style scoped>
.chat-panel {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 56px);
  background: #0f0f23;
  border-left: 1px solid #2a2a4a;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: #1a1a2e;
  border-bottom: 1px solid #2a2a4a;
}

.chat-title {
  color: #00d4ff;
  font-weight: bold;
  font-size: 0.9em;
}

.model-select {
  background: #16213e;
  color: #e0e0e0;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  padding: 2px 6px;
  font-size: 0.75em;
  font-family: monospace;
  max-width: 160px;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.chat-empty {
  color: #555;
  text-align: center;
  margin-top: 40px;
  font-style: italic;
  font-size: 0.85em;
}

.chat-loading {
  color: #00d4ff;
  font-style: italic;
  font-size: 0.8em;
  padding: 4px 12px;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.chat-input-area {
  display: flex;
  padding: 8px;
  gap: 8px;
  border-top: 1px solid #2a2a4a;
  background: #1a1a2e;
}

.chat-input {
  flex: 1;
  background: #16213e;
  color: #e0e0e0;
  border: 1px solid #2a2a4a;
  border-radius: 6px;
  padding: 8px;
  font-family: monospace;
  font-size: 0.85em;
  resize: none;
  outline: none;
}

.chat-input:focus {
  border-color: #00d4ff;
}

.chat-send {
  background: #00d4ff;
  color: #0f0f23;
  border: none;
  border-radius: 6px;
  padding: 8px 16px;
  font-weight: bold;
  font-family: monospace;
  cursor: pointer;
}

.chat-send:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.chat-send:hover:not(:disabled) {
  opacity: 0.85;
}
</style>
