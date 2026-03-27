<template>
  <div class="code-view">
    <div v-if="html" class="source-container" v-html="html"></div>
    <div v-else class="code-empty">Select a tensorBASIC program to view source.</div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "CodeView",
  props: {
    html: { type: String, default: "" },
    source: { type: String, default: "" },
  },
});
</script>

<style scoped>
.code-view {
  height: 100%;
}

.source-container {
  background: #1a1a2e;
  border: 1px solid #2a2a4a;
  border-radius: 8px;
  padding: 16px;
  overflow: auto;
  max-height: calc(100vh - 160px);
  font-family: "JetBrains Mono", "Fira Code", monospace;
  font-size: 0.85em;
  line-height: 1.7;
}

.code-empty {
  color: #555;
  text-align: center;
  margin-top: 40px;
  font-style: italic;
}
</style>

<style>
/* Global highlight classes (from tensorbasic_highlight.pl) */
.hl-lineno  { color: #555; margin-right: 1em; user-select: none; }
.hl-keyword { color: #ff79c6; font-weight: bold; }
.hl-comment { color: #6272a4; font-style: italic; }
.hl-string  { color: #f1fa8c; }
.hl-number  { color: #bd93f9; }
.hl-builtin { color: #50fa7b; }
.hl-ident   { color: #8be9fd; }
.hl-op      { color: #ffb86c; }
.hl-regex   { color: #ff6e6e; font-style: italic; }
.hl-interp  { color: #50fa7b; font-weight: bold; }
</style>
