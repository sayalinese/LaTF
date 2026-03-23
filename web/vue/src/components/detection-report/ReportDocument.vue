<script setup lang="ts">
import { computed } from 'vue';
import type { ForensicsReportData } from './types';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({ breaks: true, gfm: true });
const renderMd = (text: string): string => {
  try { return DOMPurify.sanitize(marked.parse(text) as string); }
  catch { return text; }
};

const props = defineProps<{
  data: ForensicsReportData;
}>();

const aiData = computed(() => props.data.aiDetection);

const formatTime = (isoString: string) => {
  if (!isoString) return '';
  const date = new Date(isoString);
  return date.toLocaleString('zh-CN', { hour12: false });
};
</script>

<template>
  <div class="report-document" id="report-document-content">
    <!-- Watermark -->
    <div class="watermark">潜迹寻真</div>

    <!-- Header -->
    <header class="report-header">
      <div class="header-top">
        <div class="logo">
          <i class="fa-solid fa-shield-halved"></i>
          <span>潜迹寻真 智能取证系统</span>
        </div>
        <div class="report-id">报告编号: {{ data.reportId }}</div>
      </div>
      <h1 class="report-title">数字图像真实性鉴定报告</h1>
      <div class="header-meta">
        <span>生成时间: {{ formatTime(data.timestamp) }}</span>
        <span v-if="data.platform">关联平台: {{ data.platform.toUpperCase() }}</span>
      </div>
    </header>

    <div class="report-content">
      <!-- Section 1: Material -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">1</span> 检材信息</h2>
        <div class="section-body info-grid">
          <div class="info-item">
            <label>检材缩略图</label>
            <div class="thumbnail-box">
              <img :src="data.targetImageUrl" alt="检材原图" crossorigin="anonymous" />
            </div>
          </div>
          <div class="info-item" v-if="data.sessionContext">
            <label>提取来源</label>
            <div class="context-info">
              <p><strong>会话 ID:</strong> {{ data.sessionContext.sessionId }}</p>
              <p v-if="data.sessionContext.senderRole !== 'unknown'">
                <strong>上传方身份:</strong> 
                <span class="role-badge" :class="data.sessionContext.senderRole">
                  {{ data.sessionContext.senderRole === 'buyer' ? '买家' : '卖家' }}
                </span>
              </p>
            </div>
          </div>
        </div>
      </section>

      <!-- Section 2: Conclusion -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">2</span> 鉴定结论</h2>
        <div class="section-body">
          <div class="conclusion-box" :class="aiData.isFake ? 'fake' : 'real'">
            <div class="verdict">
              <i :class="aiData.isFake ? 'fa-solid fa-triangle-exclamation' : 'fa-solid fa-circle-check'"></i>
              综合判定为：<strong>{{ aiData.isFake ? 'AIGC 生成图像 (存在篡改/伪造)' : '真实图像 (未检出明显伪造)' }}</strong>
            </div>
            <div class="confidence-meter">
              <span>置信度：{{ (aiData.confidence * 100).toFixed(2) }}%</span>
              <div class="bar-bg">
                <div class="bar-fill" :style="{ width: `${aiData.confidence * 100}%`, backgroundColor: aiData.isFake ? '#ef4444' : '#10b981' }"></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Section 3: Visual Evidence -->
      <section class="report-section" v-if="aiData.heatmapBase64">
        <h2 class="section-title"><span class="bg-icon">3</span> 视觉取证分析</h2>
        <div class="section-body">
          <p class="desc-text">运用深度学习频域特征提取模型，对图像的纹理一致性和拼接痕迹进行热力学分析。高亮区域代表极高概率的生成/篡改痕迹。</p>
          <div class="heatmap-comparison">
            <div class="img-wrap">
              <span class="img-label">原始检材</span>
              <img :src="data.targetImageUrl" alt="原图" crossorigin="anonymous"/>
            </div>
            <div class="img-wrap">
              <span class="img-label">伪影热力图</span>
              <img :src="'data:image/jpeg;base64,' + aiData.heatmapBase64" alt="热力图" />
            </div>
          </div>
        </div>
      </section>

      <!-- Section 4: Multimodal Analysis -->
      <section class="report-section" v-if="data.vlReport">
        <h2 class="section-title"><span class="bg-icon">4</span> 视觉破绽解析 (多模态)</h2>
        <div class="section-body">
          <div class="markdown-body vl-report-content" v-html="renderMd(data.vlReport)"></div>
        </div>
      </section>

      <!-- Section 5: Risky Statements -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">5</span> 对方违规风险言论取证</h2>
        <div class="section-body" v-if="data.riskyStatements && data.riskyStatements.length > 0">
          <table class="risk-table">
            <thead>
              <tr>
                <th style="width: 100px;">风险等级</th>
                <th>原话引用</th>
                <th>AI分析点评</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(item, idx) in data.riskyStatements" :key="idx">
                <td :class="['risk-badge', item.riskLevel.toLowerCase()]">{{ item.riskLevel }}</td>
                <td class="quote">
                  <div v-if="item.context" class="quote-context">{{ item.context }}</div>
                  <div>「{{ item.quote }}」</div>
                </td>
                <td class="comment">{{ item.comment }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="section-body empty-hint" v-else>
          <p style="color: #94a3b8; text-align: center; padding: 20px 0;"><i class="fa-solid fa-circle-info"></i> 未检测到对方违规风险言论，或 LLM 分析服务暂不可用。</p>
        </div>
      </section>

      <!-- Section 6: Rule Violations -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">6</span> 平台风控违规判定</h2>
        <div class="section-body" v-if="data.ruleViolations && data.ruleViolations.length > 0">
          <ul class="rule-list">
            <li v-for="(rule, idx) in data.ruleViolations" :key="idx" class="rule-item">
              <div class="rule-header"><i class="fa-solid fa-gavel"></i> <strong>{{ rule.ruleName }}</strong></div>
              <div class="rule-body">{{ rule.explanation }}</div>
            </li>
          </ul>
        </div>
        <div class="section-body empty-hint" v-else>
          <p style="color: #94a3b8; text-align: center; padding: 20px 0;"><i class="fa-solid fa-circle-info"></i> 未检测到平台规则违规项，或 LLM 分析服务暂不可用。</p>
        </div>
      </section>

      <!-- Section 7: Chat Log -->
      <section class="report-section" v-if="data.chatLog && data.chatLog.length > 0">
        <h2 class="section-title"><span class="bg-icon">7</span> 聊天记录摘录</h2>
        <div class="section-body">
          <div class="chat-log">
            <div v-for="(line, idx) in data.chatLog" :key="idx" :class="['chat-line', line.side === '我方' ? 'buyer' : 'seller']">
              <span class="chat-time">{{ line.time }}</span>
              <span class="chat-role">{{ line.side }}({{ line.role }})</span>
              <span class="chat-content">{{ line.content }}</span>
            </div>
          </div>
        </div>
      </section>

      <!-- Section 8: Action Suggestions -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">8</span> 维权操作建议</h2>
        <div class="section-body" v-if="data.actionSuggestions && data.actionSuggestions.length > 0">
          <ol class="action-list">
            <li v-for="(item, idx) in data.actionSuggestions" :key="idx" class="action-item">
              <strong>{{ item.action }}</strong>
              <p>{{ item.detail }}</p>
            </li>
          </ol>
        </div>
        <div class="section-body empty-hint" v-else>
          <p style="color: #94a3b8; text-align: center; padding: 20px 0;"><i class="fa-solid fa-circle-info"></i> 维权建议生成中或 LLM 分析服务暂不可用。</p>
        </div>
      </section>
    </div>

    <!-- Footer -->
    <footer class="report-footer">
      <div class="footer-line"></div>
      <p>⚠️ <strong>声明：</strong> 本报告由「潜迹寻真」AI 系统自动化生成，旨在辅助交易纠纷研判。平台及仲裁人员应结合其他交易证据综合定责，系统不对最终裁决承担法律责任。</p>
      <div class="signature">
        <span class="sign-label">潜迹寻真 数字签名：</span>
        <span class="sign-hash">QJ-X-{{ Date.now().toString(16).toUpperCase() }}-{{ data.reportId.split('-').pop() }}</span>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.report-document {
  position: relative;
  background: white;
  color: #1f2937;
  width: 210mm; /* A4 width roughly */
  min-height: 297mm; /* A4 height */
  margin: 0 auto;
  padding: 20mm 15mm;
  box-sizing: border-box;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0,0,0,0.05); /* Only visible on screen */
}

/* Print Specific Styles */
@media print {
  @page {
    size: A4;
    margin: 0;
  }
  .report-document {
    width: 100%;
    min-height: auto;
    box-shadow: none;
    padding: 15mm;
  }
  .watermark {
    /* Sometimes browsers hide bg graphics on print, need color-adjust */
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
}

.watermark {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%) rotate(-30deg);
  font-size: 100px;
  color: rgba(226, 232, 240, 0.4);
  font-weight: 900;
  white-space: nowrap;
  pointer-events: none;
  z-index: 0;
  user-select: none;
}

.report-header {
  border-bottom: 3px solid #1e3a8a;
  padding-bottom: 20px;
  margin-bottom: 30px;
  position: relative;
  z-index: 1;
}

.header-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
}

.logo {
  font-size: 20px;
  font-weight: bold;
  color: #1e3a8a;
  display: flex;
  align-items: center;
  gap: 10px;
}

.report-id {
  font-family: monospace;
  color: #64748b;
  font-size: 14px;
}

.report-title {
  text-align: center;
  font-size: 32px;
  letter-spacing: 4px;
  margin: 0 0 20px 0;
  color: #0f172a;
}

.header-meta {
  display: flex;
  justify-content: center;
  gap: 30px;
  color: #475569;
  font-size: 14px;
}

.report-section {
  margin-bottom: 40px;
  position: relative;
  z-index: 1;
  page-break-inside: avoid;
}

.section-title {
  font-size: 20px;
  color: #1e3a8a;
  border-bottom: 1px solid #cbd5e1;
  padding-bottom: 8px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.bg-icon {
  background: #1e3a8a;
  color: white;
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  font-size: 14px;
}

.section-body {
  padding-left: 34px;
}

.info-grid {
  display: flex;
  gap: 40px;
}

.info-item label {
  display: block;
  font-weight: 600;
  color: #475569;
  margin-bottom: 10px;
  font-size: 14px;
}

.thumbnail-box {
  width: 150px;
  height: 150px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f8fafc;
}

.thumbnail-box img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.context-info p {
  margin: 5px 0;
  font-size: 15px;
}

.role-badge {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: bold;
}
.role-badge.buyer { background: #dbeafe; color: #1e40af; }
.role-badge.seller { background: #fef3c7; color: #b45309; }

.conclusion-box {
  padding: 24px;
  border-radius: 8px;
  border-left: 6px solid #94a3b8;
  background: #f8fafc;
}
.conclusion-box.fake {
  border-left-color: #ef4444;
  background: #fef2f2;
}
.conclusion-box.real {
  border-left-color: #10b981;
  background: #ecfdf5;
}

.verdict {
  font-size: 20px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.conclusion-box.fake .verdict { color: #b91c1c; }
.conclusion-box.real .verdict { color: #047857; }

.confidence-meter {
  font-size: 14px;
  color: #475569;
}
.bar-bg {
  height: 10px;
  background: #e2e8f0;
  border-radius: 5px;
  margin-top: 8px;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.5s ease;
}

.desc-text {
  color: #64748b;
  font-size: 14px;
  margin-bottom: 20px;
  line-height: 1.6;
}

.heatmap-comparison {
  display: flex;
  gap: 20px;
  justify-content: center;
}
.img-wrap {
  text-align: center;
  flex: 1;
}
.img-wrap img {
  max-width: 100%;
  max-height: 350px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.img-label {
  display: block;
  margin-bottom: 10px;
  font-weight: 600;
  color: #334155;
}

.vl-report-content {
  font-size: 15px;
  line-height: 1.8;
  color: #334155;
  background: #f8fafc;
  padding: 20px;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.report-footer {
  margin-top: 60px;
  padding-top: 20px;
  font-size: 12px;
  color: #94a3b8;
  position: relative;
  z-index: 1;
}
.footer-line {
  height: 1px;
  background: repeating-linear-gradient(to right, #cbd5e1 0, #cbd5e1 5px, transparent 5px, transparent 10px);
  margin-bottom: 20px;
}
.report-footer p {
  margin: 0 0 10px 0;
  line-height: 1.5;
}
.signature {
  text-align: right;
  margin-top: 20px;
  font-family: monospace;
}
.sign-hash {
  color: #64748b;
  background: #f1f5f9;
  padding: 4px 8px;
  border-radius: 4px;
}

.risk-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}
.risk-table th, .risk-table td {
  border: 1px solid #e2e8f0;
  padding: 8px 12px;
  text-align: left;
}
.risk-table th {
  background: #f8fafc;
  font-weight: 600;
  color: #334155;
}
.risk-badge {
  font-weight: bold;
  text-align: center;
  border-radius: 4px;
}
.risk-badge.high { color: #dc2626; background: #fee2e2; }
.risk-badge.medium { color: #f59e0b; background: #fef3c7; }
.risk-badge.low { color: #16a34a; background: #dcfce7; }
.quote-context {
  font-size: 12px;
  color: #64748b;
  margin-bottom: 4px;
  padding-left: 8px;
  border-left: 2px solid #cbd5e1;
}
.rule-list {
  list-style: none;
  padding: 0;
  margin-top: 10px;
}
.rule-item {
  background: #f1f5f9;
  border-left: 4px solid #3b82f6;
  padding: 12px;
  margin-bottom: 10px;
  border-radius: 0 6px 6px 0;
}
.rule-header {
  color: #1e3a8a;
  margin-bottom: 6px;
  font-size: 15px;
}
.rule-body {
  color: #475569;
  font-size: 14px;
}

/* Chat Log */
.chat-log {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px;
  max-height: 400px;
  overflow-y: auto;
  font-size: 13px;
}
.chat-line {
  padding: 6px 8px;
  border-radius: 4px;
  margin-bottom: 4px;
  display: flex;
  gap: 8px;
  align-items: baseline;
}
.chat-line.buyer { background: #eff6ff; }
.chat-line.seller { background: #fef2f2; }
.chat-time { color: #94a3b8; font-size: 12px; flex-shrink: 0; min-width: 40px; }
.chat-role { color: #64748b; font-weight: 600; font-size: 12px; flex-shrink: 0; }
.chat-content { color: #1e293b; word-break: break-all; }

/* Action Suggestions */
.action-list {
  padding-left: 20px;
  margin-top: 10px;
}
.action-item {
  margin-bottom: 12px;
  line-height: 1.6;
}
.action-item strong {
  color: #1e3a8a;
  font-size: 15px;
}
.action-item p {
  color: #475569;
  font-size: 14px;
  margin: 4px 0 0 0;
}
</style>
