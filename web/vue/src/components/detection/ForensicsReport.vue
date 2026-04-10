<script setup lang="ts">
/**
 * ForensicsReport.vue
 * 交易纠纷物证鉴定报告页面 - 完整版
 * 
 * 功能：
 * 1. 会话信息（平台、买卖双方、纠纷标题）
 * 2. 消息上下文展示
 * 3. 多图片鉴定结果对比
 * 4. 三图对比展示（原图、LaRE热力图、定位热力图）
 * 5. 检测结论摘要表格
 * 6. 定位物证分析详情
 * 7. Qwen-VL逻辑审计结论
 * 8. 违规条款匹配（平台规则）
 */

import { ref, computed, onMounted } from 'vue';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

interface SessionInfo {
    id: string;
    title: string;
    buyerId: string;
    sellerId: string;
    buyerName?: string;
    sellerName?: string;
}

interface DetectedMessage {
    id: string;
    content: string;  // 图片URL
    sender: string;   // 发送者名称
    role: string;     // buyer/seller
    senderId: string;
    detectData: {
        class_name: string;
        confidence: number;
        class_idx: number;
        probabilities: Record<string, number>;
        heatmap?: string | null;
        cascade_info?: {
            global_prob: number;
            local_prob: number | null;
            crop_bbox: [number, number, number, number] | null;
        };
    } | null;
    vlReport?: string | null;
}

interface Message {
    id: string;
    role: string;
    sender_id: string;
    name: string;
    content: string;
    type: string;
    hasBeenDetected?: boolean;
}

const props = defineProps<{
    // 会话信息
    sessionInfo?: SessionInfo | null;
    // 已鉴定的消息列表
    detectedMessages?: DetectedMessage[] | null;
    // 所有消息
    allMessages?: Message[] | null;
    // 当前检测结果
    currentDetectResult?: any | null;
    // 当前VLM报告
    currentVlReport?: string | null;
    // 兼容旧接口：原始图片
    originalImage?: string | null;
    // 兼容旧接口：LaRE热力图
    lareHeatmap?: string | null;
    // 定位热力图
    localizationHeatmap?: string | null;
    // 兼容旧接口：检测结果
    detectionResult?: any | null;
    // 兼容旧接口：VLM报告
    vlmReport?: string | null;
    // 报告编号
    reportId?: string;
}>();

const emit = defineEmits<{
    (e: 'close'): void;
}>();

// 当前活跃的标签页
const activeTab = ref<'overview' | 'messages' | 'analysis' | 'rules' | 'evidence'>('overview');

// 是否显示VLM思维链
const showThink = ref(false);

// 当前选中的鉴定图片索引
const selectedDetectionIndex = ref(0);

// 生成时间
const reportDate = computed(() => {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}`;
});

// 报告编号
const reportNumber = computed(() => {
    if (props.reportId) return props.reportId;
    const now = new Date();
    const timestamp = String(now.getTime()).slice(-6);
    return `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}-${timestamp}`;
});

// 获取当前选中的鉴定结果
const currentDetection = computed(() => {
    // 优先使用detectedMessages
    if (props.detectedMessages && props.detectedMessages.length > 0) {
        if (selectedDetectionIndex.value < props.detectedMessages.length) {
            return props.detectedMessages[selectedDetectionIndex.value];
        }
        return props.detectedMessages[0];
    }
    // 兼容旧接口
    return props.currentDetectResult || props.detectionResult;
});

// 获取当前选中的图片URL
const currentImageUrl = computed(() => {
    const detection = currentDetection.value;
    if (detection?.content) {
        return detection.content;
    }
    return props.originalImage || null;
});

// 获取当前选中的LaRE热力图
const currentLareHeatmap = computed(() => {
    const detection = currentDetection.value;
    if (detection?.detectData?.heatmap) {
        return detection.detectData.heatmap;
    }
    return props.lareHeatmap || props.currentDetectResult?.heatmap || null;
});

// 获取当前选中的定位热力图
const currentLocalizationHeatmap = computed(() => {
    return props.localizationHeatmap || null;
});

// 获取当前VLM报告
const currentVlReport = computed(() => {
    const detection = currentDetection.value;
    if (detection?.vlReport) {
        return detection.vlReport;
    }
    return props.currentVlReport || props.vlmReport || null;
});

// 是否为AI生成判定
const isFake = computed(() => {
    const detection = currentDetection.value;
    if (detection?.detectData) {
        return detection.detectData.class_idx === 1;
    }
    if (detection?.class_idx !== undefined) {
        return detection.class_idx === 1;
    }
    return props.detectionResult?.class_idx === 1 || false;
});

// 置信度
const confidence = computed(() => {
    const detection = currentDetection.value;
    if (detection?.detectData) {
        return detection.detectData.confidence;
    }
    if (detection?.confidence !== undefined) {
        return detection.confidence;
    }
    return props.detectionResult?.confidence || 0;
});

// 置信度百分比
const confidencePercent = computed(() => {
    return (confidence.value * 100).toFixed(2);
});

// 真实概率
const realPercent = computed(() => {
    const detection = currentDetection.value;
    let probs = null;
    if (detection?.detectData?.probabilities) {
        probs = detection.detectData.probabilities;
    } else if (detection?.probabilities) {
        probs = detection.probabilities;
    } else if (props.detectionResult?.probabilities) {
        probs = props.detectionResult.probabilities;
    }
    if (probs) {
        const realKey = Object.keys(probs).find(k => k.toLowerCase().includes('real')) || 'Real Photo';
        return ((probs[realKey] || 0) * 100).toFixed(2);
    }
    return '0.00';
});

// AI概率
const aiPercent = computed(() => {
    const detection = currentDetection.value;
    let probs = null;
    if (detection?.detectData?.probabilities) {
        probs = detection.detectData.probabilities;
    } else if (detection?.probabilities) {
        probs = detection.probabilities;
    } else if (props.detectionResult?.probabilities) {
        probs = props.detectionResult.probabilities;
    }
    if (probs) {
        const aiKey = Object.keys(probs).find(k => k.toLowerCase().includes('ai')) || 'AI Generated';
        return ((probs[aiKey] || 0) * 100).toFixed(2);
    }
    return '0.00';
});

// 解析VLM报告为HTML
const parsedVlmReport = computed(() => {
    const report = currentVlReport.value;
    if (!report) return null;
    
    // 提取思维链
    const thinkRegex = /<think>([\s\S]*?)<\/think>/;
    const match = report.match(thinkRegex);
    let thinkText = null;
    let mainContent = report;

    if (match) {
        thinkText = match[1].trim();
        mainContent = report.replace(thinkRegex, '').trim();
    }
    
    // Markdown 渲染
    const rawHtml = marked.parse(mainContent) as string;
    const cleanHtml = DOMPurify.sanitize(rawHtml);

    return {
        think: thinkText,
        htmlContent: cleanHtml
    };
});

// 违规条款匹配
const matchedRules = computed(() => {
    if (!isFake.value) {
        return [];
    }
    
    const conf = confidence.value;
    const severity = conf > 0.9 ? '高' : conf > 0.7 ? '中' : '低';
    
    return [
        {
            id: 'TR-001',
            content: '禁止发布AI生成的虚假商品图片，误导消费者购买决策',
            severity: severity,
            match: 'AI生成图像检测'
        },
        {
            id: 'TR-015',
            content: '不得使用图像编辑手段伪造交易凭证、评价截图等重要证据',
            severity: severity === '高' ? '中' : '低',
            match: `检测置信度 ${confidencePercent.value}%`
        },
        {
            id: 'TR-022',
            content: '禁止上传含有虚假宣传内容的图像素材',
            severity: severity === '高' ? '高' : '低',
            match: '视觉逻辑分析支持'
        }
    ];
});

// 定位分析数据
const localizationAnalysis = computed(() => {
    const detection = currentDetection.value;
    let cascadeInfo = null;
    
    if (detection?.detectData?.cascade_info) {
        cascadeInfo = detection.detectData.cascade_info;
    }
    
    if (!cascadeInfo?.crop_bbox) {
        return {
            region: '未检测到明显篡改区域',
            type: '未检出篡改',
            confidence: 0
        };
    }
    
    const [x1, y1, x2, y2] = cascadeInfo.crop_bbox;
    const width = x2 - x1;
    const height = y2 - y1;
    
    return {
        region: `[${x1}, ${y1}, ${x2}, ${y2}]`,
        dimensions: `${width} x ${height}`,
        type: '局部替换/合成',
        confidence: cascadeInfo.global_prob
    };
});

// 买卖双方信息
const partyInfo = computed(() => {
    const info = props.sessionInfo || {};
    return {
        buyer: {
            id: info.buyerId || '未知',
            name: info.buyerName || '买家',
            role: 'buyer'
        },
        seller: {
            id: info.sellerId || '未知',
            name: info.sellerName || '卖家',
            role: 'seller'
        }
    };
});

// 消息时间线
const messageTimeline = computed(() => {
    if (!props.allMessages) return [];
    return props.allMessages.slice(-10).map(m => ({
        ...m,
        time: m.content?.created_at || new Date().toISOString(),
        isDetected: m.hasBeenDetected
    }));
});

// 鉴定统计
const detectionStats = computed(() => {
    const messages = props.detectedMessages || [];
    if (messages.length === 0) {
        return {
            total: 0,
            fake: 0,
            real: 0,
            fakeRate: '0%'
        };
    }
    
    const fake = messages.filter(m => m.detectData?.class_idx === 1).length;
    const real = messages.length - fake;
    
    return {
        total: messages.length,
        fake,
        real,
        fakeRate: `${((fake / messages.length) * 100).toFixed(1)}%`
    };
});

// 模型版本信息
const modelInfo = computed(() => ({
    latf: 'LaTF v13',
    lare: 'SDXL',
    localization: 'V14 FLH',
    clip: 'RN50x64'
}));
</script>

<template>
    <div class="forensics-report">
        <!-- 报告头部 -->
        <header class="report-header">
            <div class="header-main">

                <h1 class="report-title">交易纠纷物证鉴定报告</h1>
            </div>
            <div class="header-meta">
                <div class="meta-item">
                    <span class="meta-label">报告编号</span>
                    <span class="meta-value">{{ reportNumber }}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">生成时间</span>
                    <span class="meta-value">{{ reportDate }}</span>
                </div>
 
            </div>
        </header>

        <!-- 会话信息卡片 -->
        <section class="session-info-card" v-if="sessionInfo">
            <div class="session-header">
                <i class="fa-solid fa-gavel"></i>
                <span>纠纷会话信息</span>
            </div>
            <div class="session-content">
                <div class="session-title">
                    <span class="label">纠纷标题</span>
                    <span class="value">{{ sessionInfo.title || '未命名纠纷' }}</span>
                </div>
                <div class="parties-info">
                    <div class="party buyer">
                        <div class="party-icon"><i class="fa-solid fa-user"></i></div>
                        <div class="party-details">
                            <span class="party-role">买家（申诉方）</span>
                            <span class="party-name">{{ partyInfo.buyer.name }}</span>
                        </div>
                    </div>
                    <div class="vs-divider">VS</div>
                    <div class="party seller">
                        <div class="party-icon"><i class="fa-solid fa-store"></i></div>
                        <div class="party-details">
                            <span class="party-role">卖家（被申诉方）</span>
                            <span class="party-name">{{ partyInfo.seller.name }}</span>
                        </div>
                    </div>
                </div>
                <div class="session-id">
                    <span class="label">会话ID</span>
                    <span class="value code">{{ sessionInfo.id }}</span>
                </div>
            </div>
        </section>

        <!-- 鉴定统计 -->
        <section class="stats-section">
            <div class="stats-header">
                <i class="fa-solid fa-chart-bar"></i>
                <span>鉴定统计</span>
            </div>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-value">{{ detectionStats.total }}</span>
                    <span class="stat-label">已鉴定图片</span>
                </div>
                <div class="stat-item fake">
                    <span class="stat-value">{{ detectionStats.fake }}</span>
                    <span class="stat-label">疑似伪造</span>
                </div>
                <div class="stat-item real">
                    <span class="stat-value">{{ detectionStats.real }}</span>
                    <span class="stat-label">真实图片</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">{{ detectionStats.fakeRate }}</span>
                    <span class="stat-label">伪造率</span>
                </div>
            </div>
        </section>

        <!-- 标签页导航 -->
        <nav class="tab-nav">
            <button 
                :class="['tab-btn', { active: activeTab === 'overview' }]"
                @click="activeTab = 'overview'"
            >
                <i class="fa-solid fa-home"></i>
                鉴定总览
            </button>
            <button 
                :class="['tab-btn', { active: activeTab === 'evidence' }]"
                @click="activeTab = 'evidence'"
            >
                <i class="fa-solid fa-images"></i>
                证据图集
            </button>
            <button 
                :class="['tab-btn', { active: activeTab === 'messages' }]"
                @click="activeTab = 'messages'"
            >
                <i class="fa-solid fa-comments"></i>
                消息上下文
            </button>
            <button 
                :class="['tab-btn', { active: activeTab === 'rules' }]"
                @click="activeTab = 'rules'"
            >
                <i class="fa-solid fa-gavel"></i>
                违规条款
            </button>
        </nav>

        <!-- 标签页内容 -->
        <div class="tab-content">
            <!-- 鉴定总览 -->
            <div v-show="activeTab === 'overview'" class="tab-panel">
                <div class="overview-grid">
                    <!-- 左侧：三图对比 -->
                    <div class="comparison-section">
                        <div class="section-title">
                            <i class="fa-solid fa-images"></i>
                            图像证据对比
                        </div>
                        <div class="image-comparison">
                            <div class="comparison-grid">
                                <!-- 原始图像 -->
                                <div class="comparison-item">
                                    <div class="item-label">
                                        <i class="fa-solid fa-image"></i>
                                        <span>原始图像</span>
                                    </div>
                                    <div class="image-frame">
                                        <img 
                                            v-if="currentImageUrl" 
                                            :src="currentImageUrl" 
                                            alt="原始图像" 
                                            class="preview-img"
                                        />
                                        <div v-else class="image-placeholder">
                                            <i class="fa-solid fa-image"></i>
                                            <span>无图片</span>
                                        </div>
                                    </div>
                                </div>

                                <!-- LaRE特征热力图 -->
                                <div class="comparison-item">
                                    <div class="item-label lare">
                                        <i class="fa-solid fa-fire"></i>
                                        <span>LaRE热力图</span>
                                    </div>
                                    <div class="image-frame">
                                        <img 
                                            v-if="currentLareHeatmap" 
                                            :src="'data:image/jpeg;base64,' + currentLareHeatmap" 
                                            alt="LaRE热力图" 
                                            class="preview-img"
                                        />
                                        <div v-else class="image-placeholder">
                                            <i class="fa-solid fa-fire"></i>
                                            <span>未生成</span>
                                        </div>
                                    </div>
                                    <div class="item-desc">AI生成痕迹定位</div>
                                </div>

                                <!-- 定位物证热力图 -->
                                <div class="comparison-item">
                                    <div class="item-label localization">
                                        <i class="fa-solid fa-fingerprint"></i>
                                        <span>定位热力图</span>
                                    </div>
                                    <div class="image-frame">
                                        <img 
                                            v-if="currentLocalizationHeatmap" 
                                            :src="'data:image/jpeg;base64,' + currentLocalizationHeatmap" 
                                            alt="定位热力图" 
                                            class="preview-img"
                                        />
                                        <div v-else class="image-placeholder">
                                            <i class="fa-solid fa-fingerprint"></i>
                                            <span>未生成</span>
                                        </div>
                                    </div>
                                    <div class="item-desc">篡改区域定位</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- 右侧：检测结论 -->
                    <div class="conclusion-section">
                        <div class="section-title">
                            <i class="fa-solid fa-clipboard-check"></i>
                            检测结论
                        </div>
                        
                        <div :class="['verdict-card', isFake ? 'fake' : 'real']">
                            <div class="verdict-icon">
                                <i :class="isFake ? 'fa-solid fa-robot' : 'fa-solid fa-shield-check'"></i>
                            </div>
                            <div class="verdict-text">
                                <span class="verdict-label">{{ isFake ? '疑似伪造' : '真实图片' }}</span>
                                <span class="verdict-confidence">置信度 {{ confidencePercent }}%</span>
                            </div>
                        </div>

                        <div class="confidence-bar-wrapper">
                            <div class="confidence-labels">
                                <span>真实 0%</span>
                                <span>AI生成 100%</span>
                            </div>
                            <div class="confidence-bar-bg">
                                <div 
                                    class="confidence-bar-fill"
                                    :class="isFake ? 'fill-fake' : 'fill-real'"
                                    :style="{ width: aiPercent + '%' }"
                                ></div>
                            </div>
                            <div class="confidence-values">
                                <span class="real-value">{{ realPercent }}% 真实</span>
                                <span class="ai-value">{{ aiPercent }}% AI</span>
                            </div>
                        </div>

                        <!-- VLM报告摘要 -->
                        <div class="vlm-summary" v-if="parsedVlmReport">
                            <div class="vlm-header">
                                <i class="fa-solid fa-brain"></i>
                                <span>Qwen-VL 逻辑审计</span>
                            </div>
                            <div class="vlm-content markdown-body" v-html="parsedVlmReport.htmlContent"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 证据图集 -->
            <div v-show="activeTab === 'evidence'" class="tab-panel">
                <div class="evidence-gallery">
                    <div class="gallery-grid">
                        <div 
                            v-for="(msg, index) in (detectedMessages || [])" 
                            :key="msg.id"
                            :class="['gallery-item', { active: selectedDetectionIndex === index }]"
                            @click="selectedDetectionIndex = index"
                        >
                            <img :src="msg.content" alt="证据图片" class="gallery-thumb" />
                            <div class="gallery-overlay">
                                <span :class="['detection-badge', msg.detectData?.class_idx === 1 ? 'fake' : 'real']">
                                    {{ msg.detectData?.class_idx === 1 ? '疑似伪造' : '真实' }}
                                </span>
                                <span class="sender-badge">{{ msg.sender }}</span>
                            </div>
                        </div>
                    </div>
                    <div class="gallery-empty" v-if="!detectedMessages || detectedMessages.length === 0">
                        <i class="fa-solid fa-images"></i>
                        <p>暂无已鉴定的图片证据</p>
                    </div>
                </div>
            </div>

            <!-- 消息上下文 -->
            <div v-show="activeTab === 'messages'" class="tab-panel">
                <div class="messages-timeline">
                    <div 
                        v-for="msg in (allMessages || []).slice(-15)" 
                        :key="msg.id"
                        :class="['timeline-item', msg.role]"
                    >
                        <div class="timeline-avatar">
                            <i :class="msg.role === 'buyer' ? 'fa-solid fa-user' : 'fa-solid fa-store'"></i>
                        </div>
                        <div class="timeline-content">
                            <div class="timeline-header">
                                <span class="timeline-sender">{{ msg.name }}</span>
                                <span :class="['timeline-role', msg.role]">{{ msg.role === 'buyer' ? '买家' : '卖家' }}</span>
                                <span v-if="msg.hasBeenDetected" class="detected-tag">
                                    <i class="fa-solid fa-check-circle"></i> 已鉴定
                                </span>
                            </div>
                            <div class="timeline-message" v-if="msg.type === 'text'">
                                {{ msg.content }}
                            </div>
                            <div class="timeline-message image" v-else>
                                <img :src="msg.content" alt="消息图片" />
                            </div>
                        </div>
                    </div>
                    <div class="timeline-empty" v-if="!allMessages || allMessages.length === 0">
                        <i class="fa-solid fa-comments"></i>
                        <p>暂无消息记录</p>
                    </div>
                </div>
            </div>

            <!-- 违规条款 -->
            <div v-show="activeTab === 'rules'" class="tab-panel">
                <div class="rules-section">
                    <div class="rules-header">
                        <i class="fa-solid fa-gavel"></i>
                        <span>违规条款匹配</span>
                    </div>
                    
                    <div v-if="matchedRules.length > 0" class="rules-list">
                        <div v-for="rule in matchedRules" :key="rule.id" class="rule-item">
                            <div class="rule-header">
                                <div class="rule-id">
                                    <i class="fa-solid fa-bookmark"></i>
                                    <span>规则编号: {{ rule.id }}</span>
                                </div>
                                <div :class="['rule-severity', 'severity-' + rule.severity]">
                                    {{ rule.severity }}
                                </div>
                            </div>
                            <div class="rule-content">
                                {{ rule.content }}
                            </div>
                            <div class="rule-match">
                                <i class="fa-solid fa-link"></i>
                                <span>匹配依据: {{ rule.match }}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div v-else class="rules-empty success">
                        <i class="fa-solid fa-check-circle"></i>
                        <p>未检测到违规</p>
                        <span>该图像未触发平台规则违规条款</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- 报告声明 -->
        <footer class="report-footer">
            <div class="declaration">
                <div class="declaration-icon">
                    <i class="fa-solid fa-shield-halved"></i>
                </div>
                <div class="declaration-text">
                    <h4>检测机构声明</h4>
                    <p>本报告由自动化系统生成，仅供参考。具体定责需结合其他证据综合判断。</p>
                    <div class="tech-support">
                        <span>技术支持: {{ modelInfo.latf }}</span>
                        <span>|</span>
                        <span>LaRE: {{ modelInfo.lare }}</span>
                        <span>|</span>
                        <span>定位: {{ modelInfo.localization }}</span>
                    </div>
                </div>
            </div>
        </footer>
    </div>
</template>

<style scoped>
.forensics-report {
    background: var(--card-bg, #1a1a2e);
    border: 1px solid var(--border-color, #2d2d44);
    border-radius: 20px;
    overflow: hidden;
    max-width: 1200px;
    margin: 0 auto;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* 报告头部 */
.report-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 32px 40px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
    border-bottom: 1px solid var(--border-color, #2d2d44);
}

.header-main {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.platform-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 20px;
    color: white;
    font-size: 0.85rem;
    font-weight: 600;
    width: fit-content;
}

.report-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-main, #f1f5f9);
    margin: 0;
    letter-spacing: 0.5px;
}

.header-meta {
    display: flex;
    gap: 24px;
}

.meta-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    text-align: right;
}

.meta-label {
    font-size: 0.75rem;
    color: var(--text-muted, #94a3b8);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.meta-value {
    font-size: 0.9rem;
    color: var(--text-main, #f1f5f9);
    font-weight: 500;
    font-family: 'Consolas', monospace;
}

.meta-value.code {
    font-size: 0.75rem;
    word-break: break-all;
}

/* 会话信息卡片 */
.session-info-card {
    padding: 24px 40px;
    border-bottom: 1px solid var(--border-color, #2d2d44);
}

.session-header, .stats-header, .rules-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-main, #f1f5f9);
    margin-bottom: 16px;
}

.session-header i, .stats-header i, .rules-header i {
    color: #6366f1;
}

.session-content {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.session-title {
    display: flex;
    align-items: center;
    gap: 16px;
}

.session-title .label {
    font-size: 0.85rem;
    color: var(--text-muted, #64748b);
    min-width: 80px;
}

.session-title .value {
    font-size: 1rem;
    color: var(--text-main, #f1f5f9);
    font-weight: 500;
}

.parties-info {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 16px 20px;
    background: var(--border-color-light, #252538);
    border-radius: 12px;
}

.party {
    display: flex;
    align-items: center;
    gap: 12px;
    flex: 1;
}

.party-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
}

.party.buyer .party-icon {
    background: linear-gradient(135deg, rgba(96, 165, 250, 0.2), rgba(59, 130, 246, 0.2));
    color: #60a5fa;
}

.party.seller .party-icon {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.2));
    color: #fbbf24;
}

.party-details {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.party-role {
    font-size: 0.75rem;
    color: var(--text-muted, #64748b);
}

.party-name {
    font-size: 0.95rem;
    color: var(--text-main, #f1f5f9);
    font-weight: 500;
}

.vs-divider {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--text-muted, #475569);
    padding: 0 12px;
}

.session-id {
    display: flex;
    align-items: center;
    gap: 16px;
}

.session-id .label {
    font-size: 0.85rem;
    color: var(--text-muted, #64748b);
    min-width: 80px;
}

/* 统计区域 */
.stats-section {
    padding: 24px 40px;
    border-bottom: 1px solid var(--border-color, #2d2d44);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}

.stat-item {
    padding: 20px;
    background: var(--border-color-light, #252538);
    border-radius: 12px;
    text-align: center;
}

.stat-value {
    display: block;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-main, #f1f5f9);
    margin-bottom: 4px;
}

.stat-item.fake .stat-value {
    color: #f87171;
}

.stat-item.real .stat-value {
    color: #34d399;
}

.stat-label {
    font-size: 0.8rem;
    color: var(--text-muted, #64748b);
}

/* 标签页导航 */
.tab-nav {
    display: flex;
    padding: 0 40px;
    border-bottom: 1px solid var(--border-color, #2d2d44);
    gap: 4px;
}

.tab-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 16px 20px;
    background: none;
    border: none;
    color: var(--text-muted, #64748b);
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
}

.tab-btn:hover {
    color: var(--text-main, #f1f5f9);
}

.tab-btn.active {
    color: #6366f1;
    border-bottom-color: #6366f1;
}

/* 标签页内容 */
.tab-content {
    padding: 32px 40px;
    min-height: 400px;
}

.tab-panel {
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* 总览布局 */
.overview-grid {
    display: grid;
    grid-template-columns: 1fr 400px;
    gap: 32px;
}

/* 三图对比 */
.comparison-section {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.section-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-main, #f1f5f9);
}

.section-title i {
    color: #6366f1;
}

.comparison-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.comparison-item {
    display: flex;
    flex-direction: column;
}

.item-label {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: var(--border-color-light, #252538);
    border-radius: 10px 10px 0 0;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-main, #f1f5f9);
}

.item-label.lare {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
    color: #f87171;
}

.item-label.localization {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.1));
    color: #60a5fa;
}

.item-desc {
    padding: 8px 14px;
    background: var(--border-color-light, #252538);
    border-radius: 0 0 10px 10px;
    font-size: 0.75rem;
    color: var(--text-muted, #94a3b8);
    text-align: center;
    margin-bottom: 8px;
}

.image-frame {
    position: relative;
    background: var(--border-color-light, #252538);
    border-radius: 0 0 10px 10px;
    overflow: hidden;
    min-height: 160px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.preview-img {
    width: 100%;
    height: auto;
    display: block;
    max-height: 200px;
    object-fit: contain;
}

.image-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    color: var(--text-muted, #64748b);
    padding: 40px;
}

.image-placeholder i {
    font-size: 2rem;
    opacity: 0.5;
}

/* 结论区域 */
.conclusion-section {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.verdict-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px;
    border-radius: 16px;
}

.verdict-card.fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.verdict-card.real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.verdict-icon {
    width: 56px;
    height: 56px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
}

.verdict-card.fake .verdict-icon {
    background: rgba(239, 68, 68, 0.2);
    color: #ef4444;
}

.verdict-card.real .verdict-icon {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
}

.verdict-text {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.verdict-label {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-main, #f1f5f9);
}

.verdict-card.fake .verdict-label {
    color: #f87171;
}

.verdict-card.real .verdict-label {
    color: #34d399;
}

.verdict-confidence {
    font-size: 0.9rem;
    color: var(--text-muted, #64748b);
}

/* 置信度条 */
.confidence-bar-wrapper {
    padding: 16px;
    background: var(--border-color-light, #252538);
    border-radius: 12px;
}

.confidence-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--text-muted, #64748b);
    margin-bottom: 8px;
}

.confidence-bar-bg {
    width: 100%;
    height: 12px;
    background: var(--border-color, #2d2d44);
    border-radius: 6px;
    overflow: hidden;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.fill-fake {
    background: linear-gradient(90deg, #f87171, #ef4444);
}

.fill-real {
    background: linear-gradient(90deg, #34d399, #10b981);
}

.confidence-values {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    margin-top: 8px;
}

.real-value {
    color: #34d399;
}

.ai-value {
    color: #f87171;
}

/* VLM摘要 */
.vlm-summary {
    padding: 16px;
    background: var(--border-color-light, #252538);
    border-radius: 12px;
    border-left: 4px solid #8b5cf6;
}

.vlm-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #a78bfa;
    margin-bottom: 12px;
}

.vlm-content {
    font-size: 0.85rem;
    color: var(--text-main, #e2e8f0);
    line-height: 1.6;
}

.vlm-content :deep(h3) {
    font-size: 1em;
    font-weight: 600;
    color: var(--text-main, #f1f5f9);
    margin: 12px 0 8px;
}

.vlm-content :deep(h3:first-child) {
    margin-top: 0;
}

.vlm-content :deep(ul) {
    margin: 8px 0;
    padding-left: 16px;
}

.vlm-content :deep(li) {
    margin-bottom: 4px;
}

.vlm-content :deep(strong) {
    color: #60a5fa;
}

/* 证据图集 */
.evidence-gallery {
    min-height: 300px;
}

.gallery-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px;
}

.gallery-item {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    cursor: pointer;
    border: 2px solid transparent;
    transition: all 0.2s ease;
}

.gallery-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
}

.gallery-item.active {
    border-color: #6366f1;
}

.gallery-thumb {
    width: 100%;
    height: 140px;
    object-fit: cover;
    display: block;
}

.gallery-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 8px;
    background: linear-gradient(to top, rgba(0, 0, 0, 0.8), transparent);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.detection-badge {
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 6px;
}

.detection-badge.fake {
    background: rgba(239, 68, 68, 0.8);
    color: white;
}

.detection-badge.real {
    background: rgba(16, 185, 129, 0.8);
    color: white;
}

.sender-badge {
    font-size: 0.7rem;
    color: rgba(255, 255, 255, 0.8);
}

.gallery-empty, .timeline-empty, .rules-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 20px;
    text-align: center;
    color: var(--text-muted, #64748b);
}

.gallery-empty i, .timeline-empty i, .rules-empty i {
    font-size: 3rem;
    margin-bottom: 16px;
    opacity: 0.5;
}

.gallery-empty p, .timeline-empty p, .rules-empty p {
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-main, #f1f5f9);
    margin-bottom: 4px;
}

.gallery-empty span, .rules-empty span {
    font-size: 0.85rem;
}

.rules-empty.success i {
    color: #34d399;
    opacity: 1;
}

/* 消息时间线 */
.messages-timeline {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.timeline-item {
    display: flex;
    gap: 12px;
    padding: 12px 16px;
    background: var(--border-color-light, #252538);
    border-radius: 12px;
}

.timeline-avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
}

.timeline-item.buyer .timeline-avatar {
    background: linear-gradient(135deg, rgba(96, 165, 250, 0.2), rgba(59, 130, 246, 0.2));
    color: #60a5fa;
}

.timeline-item.seller .timeline-avatar {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.2), rgba(245, 158, 11, 0.2));
    color: #fbbf24;
}

.timeline-item.system .timeline-avatar {
    background: rgba(99, 102, 241, 0.2);
    color: #818cf8;
}

.timeline-content {
    flex: 1;
    min-width: 0;
}

.timeline-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}

.timeline-sender {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-main, #f1f5f9);
}

.timeline-role {
    font-size: 0.7rem;
    padding: 2px 6px;
    border-radius: 4px;
}

.timeline-item.buyer .timeline-role {
    background: rgba(96, 165, 250, 0.2);
    color: #60a5fa;
}

.timeline-item.seller .timeline-role {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
}

.detected-tag {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 0.7rem;
    color: #34d399;
}

.timeline-message {
    font-size: 0.85rem;
    color: var(--text-main, #e2e8f0);
    line-height: 1.5;
}

.timeline-message.image img {
    max-width: 200px;
    border-radius: 8px;
    margin-top: 8px;
}

/* 违规条款 */
.rules-section {
    min-height: 300px;
}

.rules-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.rule-item {
    padding: 20px;
    background: var(--border-color-light, #252538);
    border-radius: 12px;
    border-left: 4px solid #f87171;
}

.rule-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.rule-id {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-main, #f1f5f9);
}

.rule-id i {
    color: #f87171;
}

.rule-severity {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
}

.severity-high {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
}

.severity-medium {
    background: rgba(251, 191, 36, 0.2);
    color: #fbbf24;
}

.severity-low {
    background: rgba(34, 197, 94, 0.2);
    color: #4ade80;
}

.rule-content {
    font-size: 0.9rem;
    color: var(--text-main, #e2e8f0);
    line-height: 1.6;
    margin-bottom: 12px;
}

.rule-match {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: var(--text-muted, #64748b);
}

.rule-match i {
    color: #6366f1;
}

/* 报告底部 */
.report-footer {
    padding: 32px 40px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.02));
    border-top: 1px solid var(--border-color, #2d2d44);
}

.declaration {
    display: flex;
    align-items: flex-start;
    gap: 20px;
}

.declaration-icon {
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    flex-shrink: 0;
}

.declaration-icon i {
    font-size: 1.5rem;
    color: white;
}

.declaration-text h4 {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-main, #f1f5f9);
    margin: 0 0 8px 0;
}

.declaration-text p {
    font-size: 0.85rem;
    color: var(--text-muted, #94a3b8);
    line-height: 1.5;
    margin: 0 0 12px 0;
}

.tech-support {
    display: flex;
    gap: 12px;
    font-size: 0.75rem;
    color: var(--text-muted, #64748b);
}

/* 响应式 */
@media (max-width: 900px) {
    .report-header {
        flex-direction: column;
        gap: 20px;
    }
    
    .header-meta {
        width: 100%;
        justify-content: flex-start;
    }
    
    .overview-grid {
        grid-template-columns: 1fr;
    }
    
    .comparison-grid {
        grid-template-columns: 1fr;
    }
    
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .parties-info {
        flex-direction: column;
        gap: 12px;
    }
    
    .vs-divider {
        padding: 8px 0;
    }
    
    .tab-nav {
        overflow-x: auto;
        padding: 0 20px;
    }
    
    .tab-btn {
        white-space: nowrap;
    }
    
    .tab-content {
        padding: 20px;
    }
}
</style>
