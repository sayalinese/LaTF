<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  heatmapBase64?: string | null
  originalSrc?: string | null
}>()

const sliderVal = ref(50)
</script>

<template>
  <div class="heatmap-outer" v-if="heatmapBase64 && originalSrc">
    <h4><i class="fa-solid fa-layer-group"></i> 分析对比</h4>
    <p class="heatmap-desc">拖动滑块查看 AI 异常热力图。</p>

    <!-- 居中弹性盒子，作为图片的边框容器 -->
    <div class="slider-frame">
      <!-- 动态大小包裹器 -->
      <div class="comparison-container">

        <!-- 底版原图，撑开对比容器的实际宽和高 -->
        <img class="base-img" :src="originalSrc" alt="Original" />

        <!-- 热力图覆盖层 -->
        <div class="heatmap-overlay" :style="{ clipPath: 'inset(0 ' + (100 - sliderVal) + '% 0 0)' }">
          <img class="heatmap-img" :src="'data:image/jpeg;base64,' + heatmapBase64" alt="Heatmap" />
        </div>

        <!-- 拖动手柄 -->
        <input type="range" min="0" max="100" v-model="sliderVal" class="slider-input" />
        <div class="slider-handle" :style="{ left: sliderVal + '%' }"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.heatmap-outer {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.heatmap-outer h4 {
  align-self: flex-start;
  margin: 0 0 4px 0;
}

.heatmap-desc {
  align-self: flex-start;
  margin: 0 0 12px 0;
  font-size: 0.85rem;
  color: var(--text-muted, #888);
}

/* 外层边框盒子：跟随图片大小收缩，居中由父级 heatmap-outer 控制 */
.slider-frame {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 6px;
  border-radius: 14px;
  border: 1px solid var(--border-color, rgba(255,255,255,0.12));
  background: rgba(0, 0, 0, 0.04);
  max-width: 100%;
}

/* 内层边框盒子：紧贴图片，宽高由 base-img 撑开 */
.comparison-container {
  position: relative;
  display: inline-flex;
  max-width: 100%;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--border-color, rgba(255,255,255,0.15));
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

/* 原图：负责把包裹器撑开成图原本真实的宽高 */
.base-img {
  display: block;
  max-width: 100%;
  max-height: 420px;
  width: auto;
  height: auto;
}

/* 覆盖层：绝对定位并覆盖 */
.heatmap-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

/* 强制热力图在内层贴满 */
.heatmap-img {
  width: 100%;
  height: 100%;
  object-fit: fill;
  display: block;
}

.slider-input {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: ew-resize;
  z-index: 10;
  margin: 0;
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
