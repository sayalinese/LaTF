# -*- coding: utf-8 -*-
import codecs
content = '''<script setup lang="ts">
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
    
    <!-- 外层包装，提供容器居中 -->
    <div class="slider-outer-box">
      <!-- 动态大小包裹器 -->
      <div class="comparison-container" :style="{ '--slider-pos': sliderVal + '%' }">
        
        <!-- 底版原图，撑开对比容器的实际宽和高 -->
        <img class="base-img" :src="originalSrc" alt="Original" />

        <!-- 顶层热力图（绝对定位占满内层容器，强制形变与原图重合） -->
        <div class="heatmap-overlay" :style="{ clipPath: \inset(0 \% 0 0)\ }">
          <img class="heatmap-img" :src="\data:image/jpeg;base64,\\" alt="Heatmap" />
        </div>

        <!-- 拖动手柄 -->
        <input type="range" min="0" max="100" v-model="sliderVal" class="slider-input" />
        <div class="slider-handle" :style="{ left: sliderVal + '%' }"></div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.heatmap-wrapper {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.heatmap-wrapper h4, .heatmap-desc {
  align-self: flex-start;
  margin: 0 0 8px 0;
}
.heatmap-desc {
  margin-bottom: 16px;
  color: var(--text-muted, #666);
}

/* 外围限定高度框并居中里面的图片 */
.slider-outer-box {
  width: 100%;
  padding: 12px;
  height: 480px; 
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0,0,0,0.03); 
  border: 1px dashed var(--border-color, #e2e8f0);
  border-radius: 12px;
  box-sizing: border-box;
}

/* 这个容器的宽和高将被 base-img 撑开，完美契合 */
.comparison-container {
  position: relative;
  display: inline-flex;  /* 关键：行内块或内联弹性以包裹内部元素 */
  max-width: 100%;
  max-height: 100%;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}

/* 原图：负责把包裹器撑开成图原本真实的宽高，受 maxHeight 约束防止过大 */
.base-img {
  display: block;
  max-width: 100%;
  max-height: 440px; 
  width: auto;
  height: auto;
  object-fit: contain;
}

/* 覆盖层：因为外框的长宽被原图限定了，此时只需要绝对定位并覆盖它 */
.heatmap-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

/* 强制热力图在内层贴满（不管原来图多长多胖，热力图被强制拉回同样大小） */
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
'''
with codecs.open(r"d:\三创\LaRE-main\web\vue\src\components\detection\HeatmapSlider.vue", "w", "utf-8") as f:
    f.write(content)

print('VUE REWRITTEN')
