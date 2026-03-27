<template>
  <div class="chat-message" :class="message.role">
    <div class="message-role">{{ message.role === 'user' ? 'You' : 'Assistant' }}</div>
    <div class="message-content" v-html="renderedContent"></div>
    <div v-if="message.toolLog && message.toolLog.length > 0" class="tool-log">
      <div
        class="tool-log-toggle"
        @click="showTools = !showTools"
      >
        {{ showTools ? '&#9660;' : '&#9654;' }} {{ message.toolLog.length }} tool call{{ message.toolLog.length > 1 ? 's' : '' }}
      </div>
      <div v-if="showTools" class="tool-entries">
        <div v-for="(entry, i) in message.toolLog" :key="i" class="tool-entry">
          <span class="tool-name">{{ entry.tool }}</span>
          <span class="tool-args">({{ entry.args }})</span>
          <span class="tool-elapsed">{{ Math.round(entry.elapsed_ms) }}ms</span>
          <div class="tool-result">{{ entry.result }}</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed } from "vue";

export default defineComponent({
  name: "ChatMessage",
  props: {
    message: { type: Object, required: true },
  },
  setup(props) {
    const showTools = ref(false);

    const renderedContent = computed(() => {
      // Simple markdown-like rendering: code blocks and inline code
      let content = props.message.content || "";
      // Escape HTML
      content = content
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      // Code blocks
      content = content.replace(
        /```(\w*)\n([\s\S]*?)```/g,
        '<pre class="code-block"><code>$2</code></pre>'
      );
      // Inline code
      content = content.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');
      // Bold
      content = content.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      // Line breaks
      content = content.replace(/\n/g, "<br>");
      return content;
    });

    return { showTools, renderedContent };
  },
});
</script>

<style scoped>
.chat-message {
  margin-bottom: 12px;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 0.85em;
  line-height: 1.5;
}

.chat-message.user {
  background: #1a3a5c;
  margin-left: 20px;
}

.chat-message.assistant {
  background: #1a2e1a;
  margin-right: 20px;
}

.message-role {
  font-size: 0.7em;
  text-transform: uppercase;
  color: #888;
  margin-bottom: 4px;
  font-weight: bold;
}

.message-content {
  color: #e0e0e0;
  word-wrap: break-word;
}

.tool-log {
  margin-top: 8px;
  border-top: 1px solid #333;
  padding-top: 4px;
}

.tool-log-toggle {
  cursor: pointer;
  color: #888;
  font-size: 0.75em;
  user-select: none;
}

.tool-log-toggle:hover {
  color: #aaa;
}

.tool-entries {
  margin-top: 4px;
}

.tool-entry {
  background: #111;
  border-radius: 4px;
  padding: 4px 8px;
  margin-bottom: 4px;
  font-family: monospace;
  font-size: 0.8em;
}

.tool-name {
  color: #50fa7b;
  font-weight: bold;
}

.tool-args {
  color: #888;
  margin-left: 4px;
}

.tool-elapsed {
  color: #666;
  margin-left: 8px;
  font-size: 0.85em;
}

.tool-result {
  color: #aaa;
  margin-top: 2px;
  white-space: pre-wrap;
  max-height: 100px;
  overflow-y: auto;
}
</style>
