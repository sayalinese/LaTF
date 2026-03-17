<script setup lang="ts">
import { ref, computed } from 'vue';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

const props = defineProps<{
    report: string;
}>();

// 解析思维链 (think 标签) 和 渲染 markdown
const parseReport = computed(() => {
    const thinkRegex = /<think>([\s\S]*?)<\/think>/;
    const match = props.report.match(thinkRegex);
    let thinkText = null;
    let mainContent = props.report;

    if (match) {
        thinkText = match[1].trim();
        mainContent = props.report.replace(thinkRegex, '').trim();
    }
    
    // Markdown 渲染并进行安全过滤
    const rawHtml = marked.parse(mainContent) as string;
    const cleanHtml = DOMPurify.sanitize(rawHtml);

    return {
        think: thinkText,
        htmlContent: cleanHtml
    };
});

const showThink = ref(false);

const toggleThink = () => {
    showThink.value = !showThink.value;
};
</script>

<template>
  <div class="logic-report-container">
    <div class="report-header">
      <i class="fa-solid fa-microscope"></i> VLM 中层逻辑与视觉证据支持
    </div>

    <!-- 思维链展示 (有的话) -->
    <div class="think-section" v-if="parseReport.think">
      <div class="think-toggle" @click="toggleThink">
        <i class="fa-solid" :class="showThink ? 'fa-chevron-up' : 'fa-chevron-down'"></i>
        {{ showThink ? '收起思考过程' : '查看推理过程' }}
      </div>
      <div class="think-content" v-show="showThink">
        {{ parseReport.think }}
      </div>
    </div>

    <!-- 最终报告内容 (使用 Markdown 解析渲染) -->
    <div class="report-content markdown-body" v-html="parseReport.htmlContent">
    </div>
  </div>
</template>

<style scoped>
.logic-report-container {
    margin-top: 0;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    overflow: hidden;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    width: 100%;
    box-sizing: border-box;
}

.logic-report-container:hover {
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.3);
    border-color: rgba(255, 255, 255, 0.1);
}

.report-header {
    background: rgba(255, 255, 255, 0.05);
    padding: 16px 20px;
    font-weight: 700;
    color: #e2e8f0;
    font-size: 1.05em;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.report-header i {
    color: #60a5fa;
    font-size: 1.2em;
}

.think-section {
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.think-toggle {
    padding: 12px 20px;
    color: #94a3b8;
    font-size: 0.9em;
    font-weight: 500;
    cursor: pointer;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
}

.think-toggle:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #cbd5e1;
}

.think-content {
    padding: 16px 20px;
    font-size: 0.85em;
    color: #94a3b8;
    white-space: pre-wrap;
    word-break: break-word;
    background: rgba(0, 0, 0, 0.2);
    border-left: 3px solid #475569;
    margin: 0 20px 15px 20px;
    border-radius: 0 6px 6px 0;
    line-height: 1.6;
    font-family: 'Consolas', 'Monaco', monospace;
}

/* 现代优雅的 Markdown 渲染样式 */
.report-content {
    padding: 24px 28px;
    color: #cbd5e1;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.7;
    font-size: 0.95rem;
    word-break: break-word;
    overflow-wrap: break-word;
    box-sizing: border-box;
    width: 100%;
}

:deep(.markdown-body h3) {
    font-size: 1.15em;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 24px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
}

/* 给 H3 添加一个漂亮的左侧装饰线 */
:deep(.markdown-body h3::before) {
    content: '';
    display: inline-block;
    width: 4px;
    height: 1.1em;
    background-color: #60a5fa;
    margin-right: 10px;
    border-radius: 2px;
}

:deep(.markdown-body h3:first-child) {
    margin-top: 0;
}

:deep(.markdown-body p) {
    margin-bottom: 16px;
    letter-spacing: 0.2px;
    color: #cbd5e1;
}

/* 重点优化带 - 的无序列表 */
:deep(.markdown-body ul) {
    margin-top: 8px;
    margin-bottom: 20px;
    padding-left: 0;
    list-style: none; /* 移除默认圆点 */
}

:deep(.markdown-body li) {
    position: relative;
    padding-left: 24px;
    margin-bottom: 10px;
    color: #cbd5e1;
}

/* 自定义列表前面的漂亮圆角/图标标记 */
:deep(.markdown-body li::before) {
    content: '';
    position: absolute;
    left: 4px;
    top: 10px;
    width: 6px;
    height: 6px;
    background-color: #818cf8;
    border-radius: 50%;
    box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15); /* 微妙的光晕排版 */
}

:deep(.markdown-body li:last-child) {
    margin-bottom: 0;
}

/* 加粗文字样式优化 */
:deep(.markdown-body strong) {
    color: #f1f5f9;
    font-weight: 600;
    background: rgba(96, 165, 250, 0.15);
    padding: 0 4px;
    border-radius: 4px;
}
</style>