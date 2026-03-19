<script setup lang="ts">
import { ref, onMounted } from 'vue';
import UploadBox from '../components/detection/UploadBox.vue';
import HeatmapSlider from '../components/detection/HeatmapSlider.vue';
import LogicReport from '../components/detection/LogicReport.vue';
import ModelInfo from '../components/common/ModelInfo.vue';
import { fetchConfig, predictImage, explainLogicStream } from '../api';
import type { PredictResponse, ConfigResponse } from '../types/api';

const currentFile = ref<File | null>(null);
const previewUrl = ref<string | null>(null);
const isPredicting = ref(false);
const result = ref<PredictResponse | null>(null);
const errorMsg = ref<string | null>(null);

// VLM Logic State
const showVlmPanel = ref(false);
const isGeneratingLogic = ref(false);
const logicReportResult = ref<string | null>(null);
const logicErrorMsg = ref<string | null>(null);

// Config Status
const sdkConfig = ref<ConfigResponse | null>(null);
const configLoading = ref(true);
const configError = ref<string | null>(null);

const loadConfig = async () => {
    configLoading.value = true;
    configError.value = null;
    try {
        sdkConfig.value = await fetchConfig();
    } catch(e: any) {
        configError.value = e.message || 'Failed to link backend';
    } finally {
        configLoading.value = false;
    }
};

onMounted(() => {
    loadConfig();
});

const onFileSelected = (file: File) => {
    currentFile.value = file;
    previewUrl.value = URL.createObjectURL(file);
    result.value = null;
    errorMsg.value = null;
    logicReportResult.value = null;
    logicErrorMsg.value = null;
    showVlmPanel.value = false;
};

const clearSelection = () => {
    currentFile.value = null;
    previewUrl.value = null;
    result.value = null;
    errorMsg.value = null;
    logicReportResult.value = null;
    logicErrorMsg.value = null;
    showVlmPanel.value = false;
};

const handlePredict = async () => {
    if(!currentFile.value) return;
    isPredicting.value = true;
    errorMsg.value = null;
    result.value = null;
    logicReportResult.value = null;
    logicErrorMsg.value = null;
    showVlmPanel.value = false;
    
    try {
        result.value = await predictImage(currentFile.value);
    } catch(e: any) {
        errorMsg.value = e.response?.data?.error || e.message || '检测异常';
    } finally {
        isPredicting.value = false;
    }
};

// Utils to convert File to Base64
const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = error => reject(error);
    });
};

const handleExplainLogic = async () => {
    if(!currentFile.value || !result.value) return;
    
    showVlmPanel.value = true;
    isGeneratingLogic.value = true;
    logicErrorMsg.value = null;
    logicReportResult.value = ""; // 初始化为空字符串，准备拼接
    
    try {
        const base64Data = await fileToBase64(currentFile.value);
        const isFake = result.value.class_idx === 1;
        
        await explainLogicStream(base64Data, isFake, (chunk) => {
            logicReportResult.value += chunk;
        });
    } catch(e: any) {
        logicErrorMsg.value = e.response?.data?.error || e.message || '生成分析报告失败，确保 Ollama 已启动。';
        if (!logicReportResult.value) {
            logicReportResult.value = null;
        }
    } finally {
        isGeneratingLogic.value = false;
    }
};
</script>

<template>
  <div class="detection-view-wrapper">
    <div class="dv-container" :class="{ expanded: !!result }">

      <!-- 页面标题 -->
      <header class="dv-header">
        <div class="logo">
          <i class="fa-solid fa-shield-halved"></i>
          <span>AIGC 图片检测器</span>
        </div>
        <p class="subtitle">上传图片，一键检测是否为 AI 生成</p>
      </header>

      <!-- 主内容卡片 -->
      <main class="dv-main" :class="{ 'has-result': !!result }">

        <!-- 上传区 / 预览区 -->
        <section class="dv-upload-panel">

          <!-- 未选择图片 -->
          <div v-if="!previewUrl" class="upload-wrapper">
            <UploadBox @file-selected="onFileSelected" />
          </div>

          <!-- 已选择图片 -->
          <div v-else class="preview-wrapper">
            <div class="preview-img-box">
              <img :src="previewUrl" alt="预览" class="preview-img" />
              <button class="remove-btn" @click="clearSelection" v-if="!isPredicting">
                <i class="fa-solid fa-xmark"></i>
              </button>
            </div>

            <!-- 操作按钮 -->
            <div class="predict-action" v-if="!result && !isPredicting">
              <button class="primary-btn predict-btn" @click="handlePredict">
                <i class="fa-solid fa-wand-magic-sparkles"></i> 开始分析预测
              </button>
            </div>

            <!-- 加载中 -->
            <div class="predicting-state" v-if="isPredicting">
              <div class="spinner"></div>
              <p>正在执行模型推理与热力图生成…</p>
            </div>

            <!-- 错误提示 -->
            <div class="error-banner" v-if="errorMsg">
              <i class="fa-solid fa-triangle-exclamation"></i> {{ errorMsg }}
            </div>
          </div>
        </section>

        <!-- 结果区（有结果时展开） -->
        <transition name="result-slide">
          <section class="dv-result-panel" v-if="result">

            <!-- 检测结论 + 热力图 + 置信度 -->
            <div class="result-card full-width">
              <div class="result-card-header">
                <i class="fa-solid fa-chart-pie"></i> 检测结论
              </div>

              <div :class="['verdict-badge', result.class_idx === 1 ? 'fake' : 'real']">
                <i class="fa-solid" :class="result.class_idx === 1 ? 'fa-robot' : 'fa-image'"></i>
                <span>{{ result.class_idx === 1 ? 'AI 生成图像' : '真实照片' }}</span>
              </div>

              <!-- 新布局：左右分栏展示热力图详情 -->
              <div class="result-details-grid">
                <div class="heatmap-section">
                  <div class="img-center-frame">
                    <HeatmapSlider
                      v-if="result.heatmap"
                      :heatmap-base64="result.heatmap"
                      :original-src="previewUrl!"
                    />
                  </div>
                </div>

                <div class="stats-section">
                  <div class="confidence-block">
                    <div class="panel-subtitle">置信度分析</div>
                    <div class="confidence-labels">
                      <span>真实 {{ ((1 - result.confidence) * 100).toFixed(1) }}%</span>
                      <span>AI {{ (result.confidence * 100).toFixed(1) }}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                      <div class="confidence-bar-fill"
                        :class="result.class_idx === 1 ? 'fill-fake' : 'fill-real'"
                        :style="{ width: result.confidence * 100 + '%' }"
                      ></div>
                    </div>
                    <div class="confidence-value">置信度 {{ result.confidence.toFixed(4) }}</div>
                  </div>
                  
                  <div class="vlm-trigger-block">
                    <div class="panel-subtitle">深度物理逻辑鉴别</div>
                    <p class="vlm-desc">启用多模态大模型(Qwen-VL)结合特征热力图进行深度物理规律层面的逻辑诊断。此过程可能需要几十秒时间。</p>
                    <button class="vlm-btn" @click="handleExplainLogic" :disabled="isGeneratingLogic" v-if="!showVlmPanel">
                      <i class="fa-solid fa-microscope"></i> 启动深度多模态诊断
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <!-- VLM 深度分析（按需展开） -->
            <transition name="vlm-slide">
              <div class="result-card full-width vlm-card" v-if="showVlmPanel">
                <div class="result-card-header">
                  <i class="fa-solid fa-microscope"></i> 多模态深度诊断报告
                </div>

                <div class="vlm-status" v-if="isGeneratingLogic">
                  <div class="spinner"></div>
                  <p>正在解析视觉特征与物理规律，生成诊断报告中...</p>
                </div>

                <div class="vlm-error" v-if="logicErrorMsg">
                  <i class="fa-solid fa-circle-exclamation"></i> {{ logicErrorMsg }}
                </div>

                <div class="logic-report-wrap" v-if="logicReportResult">
                  <LogicReport :report="logicReportResult" />
                </div>
              </div>
            </transition>

          </section>
        </transition>
      </main>
    </div>

    <ModelInfo :config="sdkConfig" :loading="configLoading" :error="configError" @refresh="loadConfig" />
  </div>
</template>

<style scoped>
/* ── 布局容器 ── */
.detection-view-wrapper {
  width: 100%;
  min-height: 100vh;
  padding: 0 16px 60px;
  box-sizing: border-box;
}

.dv-container {
  max-width: 960px;
  margin: 60px auto 0;
}

/* ── 头部 ── */
.dv-header {
  text-align: center;
  margin-bottom: 2rem;
  animation: fadeInDown 0.7s ease;
}

/* ── 主卡片 ── */
.dv-main {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  backdrop-filter: var(--glass-blur);
  border-radius: 24px;
  padding: 40px;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
  animation: fadeInUp 0.7s ease;
  display: flex;
  flex-direction: column;
  gap: 0;
}

/* ── 上传与结果之间分隔 ── */
.dv-result-panel {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* ── 上传 / 预览面板 ── */
.dv-upload-panel {
  width: 100%;
}

.upload-wrapper, .preview-wrapper {
  max-width: 600px;
  margin: 0 auto;
}

.preview-wrapper {
  display: flex;
  flex-direction: column;
  gap: 20px;
  align-items: center;
}
.preview-img-box {
  position: relative;
  display: inline-block;
  max-width: 100%;
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.preview-img {
  max-width: 100%;
  max-height: 380px;
  width: auto;
  height: auto;
  object-fit: contain;
  display: block;
}

.predict-action { 
  width: 100%; 
  display: flex;
  justify-content: center;
}
.predict-btn {
  width: 100%;
  max-width: 320px;
  font-size: 1.05rem;
  padding: 14px 0;
  border-radius: 14px;
  letter-spacing: 0.3px;
  background: linear-gradient(135deg, var(--primary-color), #8b5cf6);
  border: none;
  font-weight: 600;
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
  transition: all 0.3s ease;
}
.predict-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
}

.predicting-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 20px;
  color: var(--text-muted);
  font-size: 0.9rem;
}

.error-banner {
  width: 100%;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.25);
  color: #f87171;
  border-radius: 10px;
  padding: 12px 16px;
  font-size: 0.9rem;
  text-align: center;
}

/* ── 结果卡片 ── */
.result-card {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  min-width: 0;
  box-shadow: inset 0 0 20px rgba(255, 255, 255, 0.02);
  transition: all 0.3s ease;
}
.result-card.full-width {
  grid-column: 1 / -1;
}

.result-card-header {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-main);
  display: flex;
  align-items: center;
  gap: 8px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-color-light);
}
.result-card-header i { color: var(--primary-color); }

/* ── 判决徽章 ── */
.verdict-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  font-size: 1.15rem;
  font-weight: 700;
  padding: 14px;
  border-radius: 12px;
}
.verdict-badge.fake {
  color: #f87171;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.2);
}
.verdict-badge.real {
  color: #34d399;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.2);
}

/* ── 新布局：详情区域 ── */
.result-details-grid {
  display: flex;
  gap: 32px;
  align-items: stretch;
  margin-top: 12px;
}
.heatmap-section {
  flex: 1 1 0;
  min-width: 0;
}
.stats-section {
  flex: 1 1 0;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 28px;
}

.panel-subtitle {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-main);
  margin-bottom: 12px;
}

.vlm-trigger-block {
  background: rgba(139, 92, 246, 0.06);
  border-radius: 12px;
  padding: 20px;
  border: 1px dashed rgba(139, 92, 246, 0.25);
  flex: 1;
}

.vlm-desc {
  font-size: 0.85rem;
  color: var(--text-muted);
  line-height: 1.5;
  margin-bottom: 16px;
}

.vlm-status {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 40px 20px;
  color: var(--text-muted);
}

.vlm-card {
  border-color: rgba(139, 92, 246, 0.3);
  box-shadow: 0 0 30px rgba(139, 92, 246, 0.05) inset;
}

/* ── 热力图 ── */
.img-center-frame {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ── 置信度条 ── */
.confidence-block { display: flex; flex-direction: column; gap: 8px; }
.confidence-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
  color: var(--text-muted);
}
.confidence-bar-bg {
  width: 100%; height: 10px;
  background: var(--border-color-light);
  border-radius: 6px;
  overflow: hidden;
  position: relative;
}
.confidence-bar-fill {
  height: 100%;
  border-radius: 6px;
  position: absolute;
  right: 0;
  transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.fill-fake { background: linear-gradient(90deg, #f87171, #ef4444); }
.fill-real { background: linear-gradient(90deg, #34d399, #10b981); }
.confidence-value {
  text-align: center;
  font-size: 0.8rem;
  color: var(--text-muted);
}

/* ── VLM 按钮 ── */
.vlm-btn {
  width: 100%;
  padding: 12px 20px;
  font-size: 0.95rem;
  font-weight: 500;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  background: linear-gradient(135deg, #8b5cf6, var(--primary-color));
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}
.vlm-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
  filter: brightness(1.1);
}
.vlm-btn:disabled {
  opacity: 0.7;
  cursor: not-allowed;
  transform: none;
}
.btn-spinner {
  width: 16px !important;
  height: 16px !important;
  border-width: 2px !important;
  border-color: rgba(255,255,255,0.3) !important;
  border-top-color: white !important;
  flex-shrink: 0;
}

.vlm-error {
  background: rgba(239, 68, 68, 0.08);
  border: 1px solid rgba(239, 68, 68, 0.2);
  color: #f87171;
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 0.85rem;
}

.logic-report-wrap { flex: 1; overflow: auto; }

.vlm-placeholder {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  color: var(--text-muted);
  border: 1px dashed var(--border-color);
  border-radius: 12px;
  padding: 30px 20px;
  min-height: 180px;
  background: rgba(0,0,0,0.08);
  opacity: 0.7;
  text-align: center;
}
.vlm-placeholder i { font-size: 2.8rem; color: #475569; }
.vlm-placeholder p { font-size: 0.88rem; line-height: 1.6; max-width: 260px; }

/* ── 适配小屏幕 ── */
@media (max-width: 768px) {
  .result-details-grid {
    flex-direction: column;
    gap: 20px;
  }
  .dv-main {
    padding: 24px 16px;
  }
}

/* ── 过渡动画 ── */
.result-slide-enter-active { transition: opacity 0.4s ease, transform 0.4s cubic-bezier(0.4,0,0.2,1); }
.result-slide-enter-from { opacity: 0; transform: translateY(16px); }

.vlm-slide-enter-active { transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
.vlm-slide-enter-from { opacity: 0; transform: translateY(-10px); max-height: 0; padding-top: 0; padding-bottom: 0; overflow: hidden; margin-top: -20px; }
</style>