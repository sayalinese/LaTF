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
};

const clearSelection = () => {
    currentFile.value = null;
    previewUrl.value = null;
    result.value = null;
    errorMsg.value = null;
    logicReportResult.value = null;
    logicErrorMsg.value = null;
};

const handlePredict = async () => {
    if(!currentFile.value) return;
    isPredicting.value = true;
    errorMsg.value = null;
    result.value = null;
    logicReportResult.value = null;
    logicErrorMsg.value = null;
    
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
  <div class="container" :style="{ maxWidth: result ? '1200px' : '540px', transition: 'max-width 0.5s ease', marginTop: '100px' }">
    <header>
        <div class="logo">
            <i class="fa-solid fa-shield-halved"></i>
            <span>AIGC 图片检测器</span>
        </div>
    </header>

    <main class="card">
      <div v-if="!previewUrl" class="upload-wrapper">
         <UploadBox @file-selected="onFileSelected" />
      </div>

      <div v-else class="preview-section" style="display: flex; flex-direction: column; align-items: center; gap: 20px;">
        <div style="position: relative; display: inline-block; max-width: 100%;">
          <img :src="previewUrl" alt="预览" style="max-width: 100%; max-height: 500px; width: auto; height: auto; object-fit: contain; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.2); display: block;">
          <button class="remove-btn" @click="clearSelection" v-if="!isPredicting" style="position: absolute; top: 10px; right: 10px; z-index: 10;">
              <i class="fa-solid fa-xmark"></i>
          </button>
        </div>
        
        <div class="action-buttons" v-if="!result && !isPredicting">
          <button class="primary-btn" @click="handlePredict" style="padding: 10px 20px; font-size: 16px; border-radius: 6px; background-color: var(--primary-color, #007bff); color: white; border: none; cursor: pointer;">
              <i class="fa-solid fa-wand-magic-sparkles"></i> 开始分析预测
          </button>
        </div>
      </div>

      <div class="result-section" v-if="isPredicting || result || errorMsg" style="margin-top: 20px;">
        
        <div class="loader" v-if="isPredicting" style="text-align: center; padding: 20px;">
            <div class="spinner"></div>
            <p style="margin-top: 10px;">正在执行模型推理与热力图生成，请稍候...</p>
        </div>

        <div class="error-box" v-if="errorMsg" style="color: red; text-align: center; padding: 20px;">
            <i class="fa-solid fa-triangle-exclamation"></i> {{ errorMsg }}
        </div>

        <div class="result-content" v-if="result" style="display: flex; flex-wrap: wrap; gap: 24px; align-items: stretch; justify-content: space-between;">
          
          <!-- 左侧：基础检测结果卡片 -->
          <div class="result-left" style="flex: 1 1 calc(50% - 12px); min-width: 320px; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 16px; padding: 25px; display: flex; flex-direction: column; justify-content: space-between; box-sizing: border-box;">
            
            <div style="flex-grow: 1; display: flex; flex-direction: column; align-items: center; width: 100%;">
                <div class="status-badge" :style="{ color: result.class_idx === 1 ? '#ef4444' : '#10b981', textAlign: 'center', fontSize: '1.2em', fontWeight: 'bold', marginBottom: '20px' }">
                    <i class="fa-solid" :class="result.class_idx === 1 ? 'fa-robot' : 'fa-check-circle'"></i>
                    <span>{{ result.class_idx === 1 ? 'AI 生成图像' : '真实照片' }}</span>
                </div>

                <div style="width: 100%; max-width: 450px;">
                    <HeatmapSlider 
                      v-if="result.heatmap" 
                      :heatmap-base64="result.heatmap" 
                      :original-src="previewUrl!" 
                    />
                </div>
            </div>

            <div class="confidence-meter" style="margin-top: 25px; width: 100%;">
                <div class="meter-labels" style="display: flex; justify-content: space-between; margin-bottom: 8px; color: #cbd5e1; font-size: 0.9em;">
                    <span>真实概率: {{ ((1 - result.confidence) * 100).toFixed(1) }}%</span>
                    <span>AI概率: {{ (result.confidence * 100).toFixed(1) }}%</span>
                </div>
                <div class="meter-bar-bg" style="width: 100%; height: 12px; background-color: rgba(255, 255, 255, 0.1); border-radius: 6px; overflow: hidden; position: relative;">
                    <div class="meter-bar-fill" :style="{ width: result.confidence * 100 + '%', height: '100%', background: result.class_idx === 1 ? 'linear-gradient(90deg, #f87171, #ef4444)' : 'linear-gradient(90deg, #34d399, #10b981)', transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)', position: 'absolute', right: 0 }"></div>
                </div>
                <div style="text-align: center; margin-top: 12px; font-size: 0.85em; color: #94a3b8;">
                    预测置信度 (Confidence): {{ result.confidence.toFixed(4) }}
                </div>
            </div>
          </div>

          <!-- 右侧：VLM 逻辑诊断卡片 -->
          <div class="result-right logic-action-section" style="flex: 1 1 calc(50% - 12px); min-width: 320px; text-align: center; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 16px; padding: 25px; display: flex; flex-direction: column; justify-content: flex-start; box-sizing: border-box;">
            
            <!-- 增加与左侧对应的隐形标题高度占位，确保底部视觉平行 -->
            <div style="margin-bottom: 20px; font-size: 1.2em; font-weight: bold; color: #e2e8f0; display: flex; align-items: center; justify-content: center; gap: 8px;">
                <i class="fa-solid fa-microscope" style="color: #818cf8;"></i> 多模态深度分析
            </div>

            <button 
                class="secondary-btn" 
                @click="handleExplainLogic" 
                :disabled="isGeneratingLogic"
                style="padding: 12px 24px; font-size: 15px; border-radius: 8px; background: linear-gradient(135deg, #6366f1, #4f46e5); color: white; border: none; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; gap: 10px; transition: all 0.2s; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3); width: 100%; max-width: 400px; margin: 0 auto;"
                onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(99, 102, 241, 0.4)';"
                onmouseout="this.style.transform='none'; this.style.boxShadow='0 4px 12px rgba(99, 102, 241, 0.3)';"
            >
              <div class="spinner" style="width: 16px; height: 16px; border-width: 2px; border-color: rgba(255,255,255,0.3); border-top-color: white;" v-if="isGeneratingLogic"></div>
              <i class="fa-solid fa-wand-magic-sparkles" v-else></i>
              {{ isGeneratingLogic ? '正在生成深度逻辑报告 (通常需要 20-40 秒)...' : '生成中层物理逻辑诊断 (Qwen-VL)' }}
            </button>

            <div class="error-box" v-if="logicErrorMsg" style="color: #ef4444; margin-top: 15px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); padding: 12px; border-radius: 8px; font-size: 14px; text-align: left;">
                <i class="fa-solid fa-circle-exclamation"></i> {{ logicErrorMsg }}
            </div>

            <div style="flex-grow: 1; margin-top: 25px; text-align: left;" v-if="logicReportResult">
              <LogicReport :report="logicReportResult" />
            </div>

            <!-- 空状态占位 -->
            <div v-if="!logicReportResult && !isGeneratingLogic && !logicErrorMsg" style="flex-grow: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; opacity: 0.6; margin-top: 25px; color: #94a3b8; border: 1px dashed rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 30px; min-height: 200px; background: rgba(0,0,0,0.1);">
                <i class="fa-brands fa-searchengin" style="font-size: 3.5em; margin-bottom: 20px; color: #64748b;"></i>
                <p style="font-size: 0.95em; max-width: 85%; line-height: 1.6; text-align: center;">点击上方按钮调用 Qwen-VL，结合特征热力图为您提供物理规律层面的深层解读。</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
  <ModelInfo :config="sdkConfig" :loading="configLoading" :error="configError" @refresh="loadConfig" />
</template>