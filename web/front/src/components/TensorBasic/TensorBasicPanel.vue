<template>
  <div class="tb-panel">
    <div class="tb-toolbar">
      <select v-model="selectedFile" class="file-select" @change="onFileChange">
        <option value="" disabled>Select a program...</option>
        <option v-for="f in files" :key="f" :value="f">{{ f }}</option>
      </select>
      <div class="tab-bar">
        <button
          v-for="tab in tabs"
          :key="tab.id"
          class="tab-btn"
          :class="{ active: activeTab === tab.id }"
          @click="setTab(tab.id)"
        >
          {{ tab.label }}
        </button>
      </div>
    </div>
    <div class="tb-content">
      <CodeView v-if="activeTab === 'source'" :html="highlightHtml" :source="sourceText" />
      <MermaidView v-else-if="activeTab === 'diagram'" :src="mermaidSrc" />
      <UmapView v-else-if="activeTab === 'umap'" :data="umapData" />
      <TrainingView v-else-if="activeTab === 'training'" />
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, computed, ref, onMounted } from "vue";
import { useStore } from "@/store/index";
import CodeView from "./CodeView.vue";
import MermaidView from "./MermaidView.vue";
import UmapView from "./UmapView.vue";
import TrainingView from "./TrainingView.vue";

export default defineComponent({
  name: "TensorBasicPanel",
  components: { CodeView, MermaidView, UmapView, TrainingView },
  setup() {
    const store = useStore();

    const tabs = [
      { id: "source", label: "Source" },
      { id: "diagram", label: "Diagram" },
      { id: "umap", label: "UMAP" },
      { id: "training", label: "Training" },
    ];

    const files = computed(() => (store.state as any).tensorbasic.files);
    const activeTab = computed(() => (store.state as any).tensorbasic.activeTab);
    const highlightHtml = computed(() => (store.state as any).tensorbasic.highlightHtml);
    const sourceText = computed(() => (store.state as any).tensorbasic.sourceText);
    const mermaidSrc = computed(() => (store.state as any).tensorbasic.mermaidSrc);
    const umapData = computed(() => (store.state as any).tensorbasic.umapData);
    const selectedFile = ref("");

    onMounted(() => {
      store.dispatch("tensorbasic/loadFiles");
    });

    const onFileChange = () => {
      if (selectedFile.value) {
        store.dispatch("tensorbasic/loadFile", selectedFile.value);
      }
    };

    const setTab = (tab: string) => {
      store.commit("tensorbasic/setActiveTab", tab);
    };

    return {
      tabs,
      files,
      activeTab,
      highlightHtml,
      sourceText,
      mermaidSrc,
      umapData,
      selectedFile,
      onFileChange,
      setTab,
    };
  },
});
</script>

<style scoped>
.tb-panel {
  height: calc(100vh - 56px);
  display: flex;
  flex-direction: column;
  background: #0f0f23;
}

.tb-toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  background: #1a1a2e;
  border-bottom: 1px solid #2a2a4a;
}

.file-select {
  background: #16213e;
  color: #e0e0e0;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  padding: 4px 8px;
  font-family: monospace;
  font-size: 0.85em;
}

.tab-bar {
  display: flex;
  gap: 2px;
}

.tab-btn {
  background: transparent;
  color: #888;
  border: 1px solid transparent;
  border-radius: 4px 4px 0 0;
  padding: 4px 16px;
  cursor: pointer;
  font-family: monospace;
  font-size: 0.8em;
  transition: all 0.15s;
}

.tab-btn:hover {
  color: #e0e0e0;
}

.tab-btn.active {
  background: #16213e;
  color: #00d4ff;
  border-color: #2a2a4a;
  border-bottom-color: #16213e;
}

.tb-content {
  flex: 1;
  overflow: auto;
  padding: 16px;
}
</style>
