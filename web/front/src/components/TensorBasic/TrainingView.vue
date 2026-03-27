<template>
  <div class="training-view">
    <div class="training-empty">
      <div class="training-icon">&#9881;</div>
      <p>Training Metrics</p>
      <p class="training-hint">Run a training loop via the chat to see loss curves, weight histograms, and gradient norms.</p>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent } from "vue";

export default defineComponent({
  name: "TrainingView",
});
</script>

<style scoped>
.training-view {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.training-empty {
  text-align: center;
  color: #555;
}

.training-icon {
  font-size: 3em;
  margin-bottom: 16px;
}

.training-hint {
  font-style: italic;
  font-size: 0.85em;
  max-width: 300px;
  margin: 0 auto;
}
</style>
