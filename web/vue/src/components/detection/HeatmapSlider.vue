<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  heatmapBase64?: string | null
  originalSrc?: string | null
}>()

const sliderVal = ref(50)
</script>

<template>
  <div class="heatmap-wrapper" v-if="heatmapBase64 && originalSrc">
    <h4><i class="fa-solid fa-layer-group"></i> 分析对比</h4>
    <p class="heatmap-desc">拖动滑块查看 AI 异常热力图。</p>
    
    <div class="comparison-container" :style="{ '--slider-pos': sliderVal + '%' }">
      <!-- Bottom Image (Heatmap) -->
      <img class="comparison-image" :src="`data:image/jpeg;base64,${heatmapBase64}`" alt="Heatmap">

      
      <!-- Top Image (Original) -->
      <div class="img-overlay" :style="{ clipPath: `inset(0 ${100 - sliderVal}% 0 0)` }">
        <img :src="originalSrc" alt="Original">

      </div>
      
      <!-- Handle -->
      <input type="range" min="0" max="100" v-model="sliderVal" class="slider-input" />
      <div class="slider-handle" :style="{ left: sliderVal + '%' }"></div>
    </div>
  </div>
</template>

<style scoped>
/* Scoped overrides to make the slider work smoothly in Vue */
.comparison-container {
  position: relative;
  width: 100%;
  aspect-ratio: auto;
  border-radius: 8px;
  background-color: transparent;
  display: flex;
  justify-content: center;
  align-items: center;
}

.comparison-image {
  width: 100%;
  height: auto;
  max-height: 450px;
  object-fit: contain;
  display: block;
}

.img-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.img-overlay img {
  width: 100%;
  height: 100%;
  max-height: 450px;
  object-fit: contain;
}

.slider-input {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: ew-resize;
  z-index: 10;
}

.slider-handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background-color: white;
  z-index: 9;
  transform: translateX(-50%);
  pointer-events: none;
}
.slider-handle::after {
  content: '< >';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: white;
  color: #333;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: bold;
}
</style>