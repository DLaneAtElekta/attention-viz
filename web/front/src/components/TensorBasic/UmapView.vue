<template>
  <div class="umap-view">
    <div v-if="data && data.length > 0" class="umap-wrapper">
      <div class="umap-controls">
        <label>
          n_neighbors:
          <input type="range" v-model.number="nNeighbors" min="2" max="50" />
          <span>{{ nNeighbors }}</span>
        </label>
        <label>
          min_dist:
          <input type="range" v-model.number="minDistInt" min="0" max="100" step="1" />
          <span>{{ (minDistInt / 100).toFixed(2) }}</span>
        </label>
        <button @click="runUmap">Re-run UMAP</button>
        <label>
          <input type="checkbox" v-model="colorByRegion" />
          Color by region
        </label>
      </div>
      <div ref="plotDiv" class="umap-plot"></div>
    </div>
    <div v-else class="umap-empty">Select a program to view UMAP visualization.</div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, watch, onMounted, nextTick } from "vue";

// Load dependencies dynamically
function loadScript(src: string): Promise<void> {
  return new Promise((resolve) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement("script");
    s.src = src;
    s.onload = () => resolve();
    document.head.appendChild(s);
  });
}

export default defineComponent({
  name: "UmapView",
  props: {
    data: { type: Array as () => any[], default: () => [] },
  },
  setup(props) {
    const plotDiv = ref<HTMLElement | null>(null);
    const nNeighbors = ref(15);
    const minDistInt = ref(10);
    const colorByRegion = ref(true);

    const runUmap = async () => {
      if (!props.data || props.data.length < 3 || !plotDiv.value) return;

      await loadScript("https://cdn.jsdelivr.net/npm/umap-js@1.4.0/lib/umap-js.min.js");
      await loadScript("https://cdn.plot.ly/plotly-2.27.0.min.js");

      const features = props.data.map((d: any) => d.features);
      const UMAP = (window as any).UMAP;
      const umap = new UMAP.UMAP({
        nNeighbors: Math.min(nNeighbors.value, features.length - 1),
        minDist: minDistInt.value / 100,
        nComponents: 2,
        spread: 1.0,
      });

      const embedding = await umap.fitAsync(features);

      const groupKey = colorByRegion.value ? "region" : "category";
      const groups: Record<string, { x: number[]; y: number[]; text: string[]; name: string }> = {};
      props.data.forEach((d: any, i: number) => {
        const key = d[groupKey] || "other";
        if (!groups[key]) groups[key] = { x: [], y: [], text: [], name: key };
        groups[key].x.push(embedding[i][0]);
        groups[key].y.push(embedding[i][1]);
        groups[key].text.push("L" + d.line + ": " + d.label);
      });

      const traces = Object.values(groups).map((g) => ({
        x: g.x, y: g.y, text: g.text, name: g.name,
        mode: "markers", type: "scatter",
        marker: { size: 8, opacity: 0.8 },
        hoverinfo: "text+name",
      }));

      const layout = {
        title: "UMAP - Program Structure",
        paper_bgcolor: "#1a1a2e",
        plot_bgcolor: "#16213e",
        font: { color: "#e0e0e0" },
        xaxis: { showgrid: false, zeroline: false, title: "" },
        yaxis: { showgrid: false, zeroline: false, title: "" },
        legend: { orientation: "h", y: -0.15 },
        margin: { l: 40, r: 20, t: 50, b: 60 },
      };

      (window as any).Plotly.newPlot(plotDiv.value, traces, layout, { responsive: true });
    };

    watch(() => props.data, async () => {
      await nextTick();
      runUmap();
    });

    onMounted(() => {
      if (props.data && props.data.length > 0) runUmap();
    });

    return { plotDiv, nNeighbors, minDistInt, colorByRegion, runUmap };
  },
});
</script>

<style scoped>
.umap-view {
  height: 100%;
}

.umap-wrapper {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.umap-controls {
  background: #1a1a2e;
  border: 1px solid #2a2a4a;
  border-radius: 8px;
  padding: 8px 16px;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  font-size: 0.8em;
  color: #888;
}

.umap-controls input[type=range] {
  width: 80px;
}

.umap-controls button {
  background: #00d4ff;
  color: #0f0f23;
  border: none;
  border-radius: 4px;
  padding: 4px 12px;
  cursor: pointer;
  font-family: monospace;
  font-weight: bold;
  font-size: 0.9em;
}

.umap-plot {
  flex: 1;
  min-height: 400px;
}

.umap-empty {
  color: #555;
  text-align: center;
  margin-top: 40px;
  font-style: italic;
}
</style>
