<template>
  <div class="mermaid-view">
    <div v-if="src" ref="mermaidContainer" class="diagram-container">
      <div v-html="renderedSvg"></div>
    </div>
    <div v-else class="diagram-empty">Select a program to view its sequence diagram.</div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, watch, onMounted } from "vue";

// Load mermaid from CDN dynamically
let mermaidLoaded = false;
function loadMermaid(): Promise<void> {
  if (mermaidLoaded) return Promise.resolve();
  return new Promise((resolve) => {
    if ((window as any).mermaid) {
      mermaidLoaded = true;
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js";
    script.onload = () => {
      (window as any).mermaid.initialize({ startOnLoad: false, theme: "dark", securityLevel: "loose" });
      mermaidLoaded = true;
      resolve();
    };
    document.head.appendChild(script);
  });
}

export default defineComponent({
  name: "MermaidView",
  props: {
    src: { type: String, default: "" },
  },
  setup(props) {
    const renderedSvg = ref("");

    const render = async () => {
      if (!props.src) {
        renderedSvg.value = "";
        return;
      }
      await loadMermaid();
      try {
        const { svg } = await (window as any).mermaid.render("mermaid-diagram", props.src);
        renderedSvg.value = svg;
      } catch (e) {
        renderedSvg.value = `<pre style="color: #ff6e6e;">${e}</pre>`;
      }
    };

    watch(() => props.src, render);
    onMounted(render);

    return { renderedSvg };
  },
});
</script>

<style scoped>
.mermaid-view {
  height: 100%;
}

.diagram-container {
  background: #1a1a2e;
  border: 1px solid #2a2a4a;
  border-radius: 8px;
  padding: 24px;
  overflow: auto;
  max-height: calc(100vh - 160px);
  text-align: center;
}

.diagram-empty {
  color: #555;
  text-align: center;
  margin-top: 40px;
  font-style: italic;
}
</style>
