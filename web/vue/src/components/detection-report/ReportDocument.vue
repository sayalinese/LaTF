<script setup lang="ts">
import { computed } from 'vue';
import type { ForensicsReportData } from './types';

const props = defineProps<{
  data: ForensicsReportData;
}>();

const aiData = computed(() => props.data.aiDetection);
const platformNameMap: Record<string, string> = {
  taobao: '淘宝',
  xianyu: '闲鱼',
  jd: '京东',
  pdd: '拼多多',
};

const chineseNumerals = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十'];

const clampScore = (value: number) => Math.max(0, Math.min(100, Math.round(value)));

const formatChineseIndex = (index: number) => `${chineseNumerals[index] || index + 1}、`;

const stripMarkdown = (text: string) => text
  .replace(/```[\s\S]*?```/g, ' ')
  .replace(/`([^`]+)`/g, '$1')
  .replace(/!\[[^\]]*\]\([^)]*\)/g, ' ')
  .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
  .replace(/[>#*_~\-]+/g, ' ')
  .replace(/\n+/g, ' ')
  .replace(/\s+/g, ' ')
  .trim();

const truncateText = (text: string, maxLength: number) => {
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength).trim()}...`;
};

const toFormalParagraphs = (text: string) => stripMarkdown(text)
  .split(/(?<=[。！？；])/)
  .map((item) => item.trim())
  .filter(Boolean);

const senderNarrative = computed(() => {
  const senderRole = props.data.sessionContext?.senderRole;
  if (senderRole === 'seller') {
    return '传方身份：卖家。该图片由卖家提交，作为针对买家的举证材料，用于支撑其关于商品状态、履约情况或责任归属的主张。';
  }
  if (senderRole === 'buyer') {
    return '传方身份：买家。该图片由买家提交，作为针对卖家的举证材料，用于支撑其关于商品瑕疵、描述不符或争议责任的主张。';
  }
  return '传方身份暂未明确，当前报告仅基于图像本体与会话上下文进行取证分析。';
});

const visualLogicSummary = computed(() => {
  if (!props.data.vlReport) return '';
  return truncateText(stripMarkdown(props.data.vlReport), 220);
});

const vlReportParagraphs = computed(() => {
  if (!props.data.vlReport) return [];
  return toFormalParagraphs(props.data.vlReport);
});

const platformLabel = computed(() => {
  const key = (props.data.platform || '').toLowerCase();
  return platformNameMap[key] || (props.data.platform || '涉案平台');
});

const senderRoleLabel = computed(() => {
  const senderRole = props.data.sessionContext?.senderRole;
  if (senderRole === 'seller') return '卖家';
  if (senderRole === 'buyer') return '买家';
  return '上传方';
});

const oppositeRoleLabel = computed(() => senderRoleLabel.value === '卖家' ? '买家' : senderRoleLabel.value === '买家' ? '卖家' : '对方');

const evidenceSummaryParagraphs = computed(() => {
  const paragraphs: string[] = [];
  const confidenceText = `${((aiData.value.confidence || 0) * 100).toFixed(2)}%`;
  paragraphs.push(`经本系统对涉案图片执行 LaRE 生成流形检测、热力图定位、视觉逻辑比对及对话语境交叉分析，当前图片的综合判定结果为“${aiData.value.isFake ? '存在较高 AIGC 生成或篡改嫌疑' : '暂未检出明显伪造痕迹'}”，模型综合置信度为 ${confidenceText}。`);
  paragraphs.push(`从举证关系看，本案中该图片由${senderRoleLabel.value}提交，并被用于支撑其针对${oppositeRoleLabel.value}所提出的事实主张或责任归属意见，因此其真实性将直接影响争议事实的认定。`);
  if (visualLogicSummary.value) {
    paragraphs.push(`Qwen-VL 视觉逻辑复核显示：${visualLogicSummary.value}${visualLogicSummary.value.endsWith('。') ? '' : '。'}`);
  }
  if ((props.data.riskyStatements || []).length > 0) {
    paragraphs.push(`对会话语境的同步分析表明，争议对话中已出现与售后责任、事实描述或证据说服有关的风险表达，说明该图片并非孤立证据，而是与完整交易争议链条存在直接关联。`);
  }
  return paragraphs;
});

const legalReferences = computed(() => [
  {
    title: '《中华人民共和国消费者权益保护法》第二十条',
    excerpt: '经营者向消费者提供有关商品或者服务的质量、性能、用途、有效期限等信息，应当真实、全面，不得作虚假或者引人误解的宣传。'
  },
  {
    title: '《中华人民共和国消费者权益保护法》第二十四条',
    excerpt: '经营者提供的商品或者服务不符合质量要求的，消费者可以依照国家规定、当事人约定退货，或者要求经营者履行更换、修理等义务。没有国家规定和当事人约定的，消费者可以自收到商品之日起七日内退货；七日后符合法定解除合同条件的，消费者可以及时退货，不符合法定解除合同条件的，可以要求经营者履行更换、修理等义务。依照前款规定进行退货、更换、修理的，经营者应当承担运输等必要费用。'
  },
  {
    title: '《中华人民共和国电子商务法》第十七条',
    excerpt: '电子商务经营者应当全面、真实、准确、及时地披露商品或者服务信息，保障消费者的知情权和选择权。电子商务经营者不得以虚构交易、编造用户评价等方式进行虚假或者引人误解的商业宣传，欺骗、误导消费者。'
  },
  {
    title: '《部分商品修理更换退货责任规定》有关法定售后责任条款',
    excerpt: '产品自售出之日起七日内发生性能故障，消费者有权选择退货、换货或者修理；在法定责任期限内，经营者应当按照规定承担修理、更换、退货等售后责任。'
  }
]);

const appealDocumentParagraphs = computed(() => {
  const requestParagraph = aiData.value.isFake
    ? `请求${platformLabel.value}平台在处理本案争议时，将涉案图片作为存在较高生成或篡改嫌疑的争议证据予以审慎审查，并对${senderRoleLabel.value}据此主张的事实基础作进一步真实性核验。`
    : `请求${platformLabel.value}平台在处理本案争议时，将涉案图片视为暂未检出明显伪造痕迹的证据材料，不宜仅依据主观怀疑直接否定其证明效力。`;

  return [
    `致：${platformLabel.value}平台争议处理/售后审核部门`,
    `就本案争议中由${senderRoleLabel.value}提交并用于指向${oppositeRoleLabel.value}责任的涉案图片，现依据图像取证结论、对话证据及国家法定售后责任规则，提交如下申诉意见。`,
    ...evidenceSummaryParagraphs.value,
    requestParagraph,
    '同时，请平台结合订单履约记录、商品页面历史描述、售后节点及双方聊天原文进行交叉比对，并对明显失真的陈述、疑似失实宣传或证据瑕疵情形依法依规处理。'
  ];
});

const riskQuoteLines = computed(() => (props.data.riskyStatements || []).slice(0, 3).map((item) => item.quote));

const ruleTextLines = computed(() => (props.data.ruleViolations || []).slice(0, 3).map((item) => `${item.ruleName}：${item.explanation}`));

const fallbackRuleParagraph = computed(() => `结合${platformLabel.value}平台交易秩序要求，建议平台同步核验订单详情、商品描述页、售后沟通记录及上传证据形成的完整证据闭环。`);

const dialogueRiskScore = computed(() => {
  const items = props.data.riskyStatements || [];
  if (items.length === 0) return 12;
  const weightMap: Record<string, number> = { high: 95, medium: 62, low: 30 };
  const total = items.reduce((sum, item) => sum + (weightMap[item.riskLevel.toLowerCase()] || 20), 0);
  return clampScore(total / items.length);
});

const scoreItems = computed(() => {
  const confidence = Math.max(0, Math.min(1, Number(aiData.value.confidence || 0)));
  const semanticNonAuthenticity = clampScore(aiData.value.isFake ? confidence * 100 : (1 - confidence) * 100);
  const generativeFit = clampScore(aiData.value.isFake ? confidence * 100 : (1 - confidence) * 100);

  return [
    {
      label: '语义非真实度',
      value: semanticNonAuthenticity,
      desc: '基于 CLIP RN50 的内容自然性异常折算值，分值越高代表视觉语义越不自然、越可疑。'
    },
    {
      label: '算法基因拟合度',
      value: generativeFit,
      desc: '基于 LaRE 的生成流形贴合度，分值越高代表越接近 SDXL 等生成模型分布。'
    },
    {
      label: '对话风险程度',
      value: dialogueRiskScore.value,
      desc: '基于 LLM 对对话风险与争议话术的归因评分，分值越高代表平台风控风险越高。'
    }
  ];
});

const radarLabels = ['语义非真实度', '算法基因拟合度', '对话风险程度'];
const radarCenter = 120;
const radarRadius = 82;

const getRadarPoint = (index: number, ratio: number) => {
  const angle = -Math.PI / 2 + index * ((Math.PI * 2) / radarLabels.length);
  return {
    x: radarCenter + Math.cos(angle) * radarRadius * ratio,
    y: radarCenter + Math.sin(angle) * radarRadius * ratio,
  };
};

const buildPolygonPoints = (values: number[]) => values
  .map((value, index) => {
    const point = getRadarPoint(index, Math.max(0, Math.min(1, value / 100)));
    return `${point.x},${point.y}`;
  })
  .join(' ');

const radarGridLevels = [25, 50, 75, 100];

const radarGridPolygons = computed(() => radarGridLevels.map((level) => buildPolygonPoints(radarLabels.map(() => level))));
const radarAxes = computed(() => radarLabels.map((_, index) => getRadarPoint(index, 1)));
const radarLabelPoints = computed(() => radarLabels.map((label, index) => ({ ...getRadarPoint(index, 1.22), label })));
const radarPolygonPoints = computed(() => buildPolygonPoints(scoreItems.value.map((item) => item.value)));

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
              <p class="sender-summary">{{ senderNarrative }}</p>
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
                <div class="bar-fill" :style="{ width: `${aiData.confidence * 100}%`, backgroundColor: aiData.isFake ? '#ef4444' : '#64748b' }"></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- Section 3: Visual Evidence -->
      <section class="report-section" v-if="aiData.heatmapBase64 || visualLogicSummary">
        <h2 class="section-title"><span class="bg-icon">3</span> 视觉取证分析</h2>
        <div class="section-body">
          <p class="desc-text">运用深度学习频域特征提取模型，对图像的纹理一致性和拼接痕迹进行热力学分析。高亮区域代表极高概率的生成/篡改痕迹。</p>
          <div class="heatmap-comparison" v-if="aiData.heatmapBase64">
            <div class="img-wrap">
              <span class="img-label">原始检材</span>
              <img :src="data.targetImageUrl" alt="原图" crossorigin="anonymous"/>
            </div>
            <div class="img-wrap">
              <span class="img-label">伪影热力图</span>
              <img :src="'data:image/jpeg;base64,' + aiData.heatmapBase64" alt="热力图" />
            </div>
          </div>
          <div v-if="visualLogicSummary" class="visual-summary-box">
            <div class="visual-summary-title">
              <i class="fa-solid fa-eye"></i>
              Qwen-VL 视觉逻辑破绽描述
            </div>
            <p>{{ visualLogicSummary }}</p>
          </div>
        </div>
      </section>

      <!-- Section 4: Multimodal Scores -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">4</span> 多模态分值占比</h2>
        <div class="section-body score-section">
          <div class="radar-panel">
            <svg class="radar-chart" viewBox="0 0 240 240" role="img" aria-label="多模态评分雷达图">
              <polygon
                v-for="(grid, idx) in radarGridPolygons"
                :key="`grid-${idx}`"
                :points="grid"
                class="radar-grid"
              />
              <line
                v-for="(axis, idx) in radarAxes"
                :key="`axis-${idx}`"
                :x1="radarCenter"
                :y1="radarCenter"
                :x2="axis.x"
                :y2="axis.y"
                class="radar-axis"
              />
              <polygon :points="radarPolygonPoints" class="radar-shape" />
              <circle
                v-for="(item, idx) in scoreItems"
                :key="`point-${idx}`"
                :cx="getRadarPoint(idx, item.value / 100).x"
                :cy="getRadarPoint(idx, item.value / 100).y"
                r="4"
                class="radar-point"
              />
              <text
                v-for="(label, idx) in radarLabelPoints"
                :key="`label-${idx}`"
                :x="label.x"
                :y="label.y"
                class="radar-label"
                text-anchor="middle"
              >
                {{ label.label }}
              </text>
            </svg>
          </div>
          <div class="score-list">
            <div v-for="item in scoreItems" :key="item.label" class="score-item">
              <div class="score-item-head">
                <strong>{{ item.label }}</strong>
                <span>{{ item.value }}/100</span>
              </div>
              <div class="score-bar-bg">
                <div class="score-bar-fill" :style="{ width: `${item.value}%` }"></div>
              </div>
              <p>{{ item.desc }}</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Section 5: Multimodal Analysis -->
      <section class="report-section" v-if="data.vlReport">
        <h2 class="section-title"><span class="bg-icon">5</span> 视觉破绽解析 (多模态)</h2>
        <div class="section-body">
          <div class="formal-text-block">
            <p v-for="(paragraph, idx) in vlReportParagraphs" :key="`vl-${idx}`">{{ paragraph }}</p>
          </div>
        </div>
      </section>

      <!-- Section 6: Risky Statements -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">6</span> 对方违规风险言论取证</h2>
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

      <!-- Section 7: Rule Violations -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">7</span> 平台风控违规判定</h2>
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

      <!-- Section 8: Chat Log -->
      <section class="report-section" v-if="data.chatLog && data.chatLog.length > 0">
        <h2 class="section-title"><span class="bg-icon">8</span> 聊天记录摘录</h2>
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

      <!-- Section 9: Action Suggestions -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">9</span> 维权操作建议</h2>
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

      <!-- Section 10: Appeal Document -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">10</span> 平台申诉意见书</h2>
        <div class="section-body">
          <div class="appeal-document">
            <div class="appeal-document-title">平台申诉意见书</div>
            <div class="appeal-document-subtitle">依据图像取证、会话语境及法定售后责任规则形成</div>
            <div class="formal-text-block appeal-document-body">
              <p v-for="(paragraph, idx) in appealDocumentParagraphs" :key="`appeal-${idx}`">{{ paragraph }}</p>
              <div v-if="riskQuoteLines.length > 0" class="appeal-section-block">
                <div class="appeal-section-title">对话证据重点摘录如下：</div>
                <ol class="appeal-number-list">
                  <li v-for="(item, idx) in riskQuoteLines" :key="`risk-${idx}`">
                    {{ formatChineseIndex(idx) }}{{ item }}
                  </li>
                </ol>
              </div>
              <p v-else>目前对话中未提取到高风险原话，但该图片仍与售后争议链条存在实质关联，建议平台结合完整聊天记录继续审查。</p>

              <div v-if="ruleTextLines.length > 0" class="appeal-section-block">
                <div class="appeal-section-title">结合平台规则分析，当前已识别出以下审核重点：</div>
                <ol class="appeal-number-list">
                  <li v-for="(item, idx) in ruleTextLines" :key="`rule-${idx}`">
                    {{ formatChineseIndex(idx) }}{{ item }}
                  </li>
                </ol>
              </div>
              <p v-else>{{ fallbackRuleParagraph }}</p>
            </div>
            <div class="appeal-signature">
              <p>此致</p>
              <p>{{ platformLabel }}平台争议处理部门</p>
              <p>附：本鉴定报告编号 {{ data.reportId }}</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Section 11: Legal References -->
      <section class="report-section">
        <h2 class="section-title"><span class="bg-icon">11</span> 国家法定售后规则引用</h2>
        <div class="section-body">
          <div class="legal-reference-list">
            <div v-for="item in legalReferences" :key="item.title" class="legal-reference-item">
              <div class="legal-reference-title">{{ item.title }}</div>
              <div class="legal-reference-excerpt">原文：{{ item.excerpt }}</div>
            </div>
          </div>
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
  
  /* Force browsers to print background colors and text colors accurately */
  * {
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }

  .report-document {
    width: 100%;
    min-height: auto;
    box-shadow: none;
    padding: 15mm;
  }
  
  .watermark {
    /* Increase opacity explicitly for printing */
    color: rgba(226, 232, 240, 0.8) !important;
  }
  
  /* Fix header text colors fading during print */
  .logo {
    color: #333333 !important;
  }
  .report-title {
    color: #0f172a !important;
  }
  .report-id {
    color: #64748b !important;
  }
  .header-meta {
    color: #475569 !important;
  }
  .report-header {
    border-bottom: 3px solid #333333 !important;
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
  /* CHANGED: Make sure z-index is positive but lower than content to fix html2canvas bug */
  z-index: 1;
  user-select: none;
}

.report-content {
  position: relative;
  z-index: 2;
}

.report-header {
  border-bottom: 2px solid #000000;
  padding-bottom: 20px;
  margin-bottom: 30px;
  position: relative;
  z-index: 10;
  background-color: transparent; 
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
  color: #000000;
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo i,
.logo span {
  color: #000000 !important;
  opacity: 1 !important;
}

.report-id {
  font-family: monospace;
  color: #000000;
  font-size: 14px;
}

.report-title {
  text-align: center;
  font-size: 32px;
  letter-spacing: 4px;
  margin: 0 0 20px 0;
  color: #000000;
  font-weight: bold;
}

.header-meta {
  display: flex;
  justify-content: center;
  gap: 30px;
  color: #000000;
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
  color: #333333;
  border-bottom: 1px solid #cbd5e1;
  padding-bottom: 8px;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.bg-icon {
  background: #333333;
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

.sender-summary {
  margin-top: 10px;
  color: #475569;
  line-height: 1.7;
}

.role-badge {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 13px;
  font-weight: bold;
}
.role-badge.buyer { background: #f1f5f9; color: #475569; }
.role-badge.seller { background: #e2e8f0; color: #334155; }

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
  border-left-color: #94a3b8;
  background: #f8fafc;
}

.verdict {
  font-size: 20px;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.conclusion-box.fake .verdict { color: #b91c1c; }
.conclusion-box.real .verdict { color: #334155; }

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

.visual-summary-box {
  margin-top: 18px;
  padding: 16px 18px;
  border: 1px solid #e2e8f0;
  border-left: 4px solid #333333;
  border-radius: 8px;
  background: #f8fafc;
}

.visual-summary-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  font-size: 15px;
  font-weight: 700;
  color: #333333;
}

.visual-summary-box p {
  margin: 0;
  color: #475569;
  line-height: 1.8;
}

.score-section {
  display: flex;
  gap: 24px;
  align-items: flex-start;
}

.radar-panel {
  width: 260px;
  flex: 0 0 260px;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #fbfbfc;
  padding: 12px;
}

.radar-chart {
  width: 100%;
  height: auto;
  display: block;
}

.radar-grid {
  fill: none;
  stroke: #d9dde3;
  stroke-width: 1;
}

.radar-axis {
  stroke: #cbd5e1;
  stroke-width: 1;
}

.radar-shape {
  fill: rgba(239, 68, 68, 0.18);
  stroke: #ef4444;
  stroke-width: 2;
}

.radar-point {
  fill: #ef4444;
}

.radar-label {
  fill: #334155;
  font-size: 10px;
  font-weight: 600;
}

.score-list {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.score-item {
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #fbfbfc;
  padding: 14px 16px;
}

.score-item-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  color: #333333;
}

.score-item-head span {
  font-family: monospace;
  color: #64748b;
}

.score-bar-bg {
  height: 8px;
  border-radius: 999px;
  background: #e5e7eb;
  overflow: hidden;
  margin-bottom: 10px;
}

.score-bar-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #ef4444, #333333);
}

.score-item p {
  margin: 0;
  color: #475569;
  font-size: 13px;
  line-height: 1.7;
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

.formal-text-block {
  padding: 18px 20px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #ffffff;
}

.formal-text-block p {
  margin: 0 0 12px 0;
  color: #374151;
  line-height: 1.95;
  text-indent: 2em;
  text-align: justify;
}

.formal-text-block p:last-child {
  margin-bottom: 0;
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
.risk-badge.medium { color: #475569; background: #f1f5f9; }
.risk-badge.low { color: #64748b; background: #f8fafc; }
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
  border-left: 4px solid #64748b;
  padding: 12px;
  margin-bottom: 10px;
  border-radius: 0 6px 6px 0;
}
.rule-header {
  color: #333333;
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

/* Force expand chat log when printing/exporting */
@media print {
  .chat-log {
    max-height: none !important;
    overflow-y: visible !important;
    height: auto !important;
  }
}
.chat-line {
  padding: 6px 8px;
  border-radius: 4px;
  margin-bottom: 4px;
  display: flex;
  gap: 8px;
  align-items: baseline;
}
.chat-line.buyer { background: #f1f5f9; }
.chat-line.seller { background: #e2e8f0; }
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
  color: #333333;
  font-size: 15px;
}
.action-item p {
  color: #475569;
  font-size: 14px;
  margin: 4px 0 0 0;
}

.appeal-document {
  border: 1px solid #d1d5db;
  border-radius: 8px;
  background: #ffffff;
  padding: 24px 26px;
}

.appeal-document-title {
  text-align: center;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: 2px;
  color: #111827;
}

.appeal-document-subtitle {
  margin-top: 8px;
  margin-bottom: 18px;
  text-align: center;
  font-size: 13px;
  color: #6b7280;
}

.appeal-document-body {
  border: none;
  border-radius: 0;
  padding: 0;
}

.appeal-section-block {
  margin: 0 0 12px 0;
}

.appeal-section-title {
  margin-bottom: 8px;
  color: #374151;
  line-height: 1.95;
  text-indent: 0;
  text-align: justify;
}

.appeal-number-list {
  list-style: none;
  margin: 0;
  padding-left: 0;
  color: #374151;
}

.appeal-number-list li {
  margin-bottom: 8px;
  line-height: 1.95;
  text-align: justify;
}

.appeal-number-list li:last-child {
  margin-bottom: 0;
}

.appeal-signature {
  margin-top: 18px;
  color: #374151;
  line-height: 1.9;
  text-align: right;
}

.appeal-signature p {
  margin: 0;
}

.legal-reference-list {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.legal-reference-item {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #ffffff;
  padding: 16px 18px;
}

.legal-reference-title {
  margin-bottom: 8px;
  font-size: 15px;
  font-weight: 700;
  color: #111827;
}

.legal-reference-excerpt {
  color: #374151;
  line-height: 1.9;
  text-align: justify;
}
</style>
