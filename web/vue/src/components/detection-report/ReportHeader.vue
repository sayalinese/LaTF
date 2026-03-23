<template>
    <div class="report-header">
        <div class="header-left">
            <div class="logo-icon"><i class="fa-solid fa-shield-halved"></i></div>
            <div class="header-info">
                <h1 class="report-title">交易纠纷物证鉴定报告</h1>
                <p class="report-subtitle">Transaction Dispute Forensics Report</p>
            </div>
        </div>
        <div class="header-right">
            <div class="meta-item"><span class="meta-label">报告编号</span><span class="meta-value">{{ reportNumber }}</span></div>
            <div class="meta-item"><span class="meta-label">生成时间</span><span class="meta-value">{{ reportDate }}</span></div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{ reportId?: string }>();

const reportDate = computed(() => {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
});

const reportNumber = computed(() => props.reportId || `${new Date().getFullYear()}${String(new Date().getMonth() + 1).padStart(2, '0')}${String(new Date().getDate()).padStart(2, '0')}-${String(new Date().getTime()).slice(-6)}`);
</script>

<style scoped>
.report-header { padding: 20px 28px; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; background: linear-gradient(135deg, #f8fafc 0%, #fff 100%); }
.header-left { display: flex; align-items: center; gap: 14px; }
.logo-icon { width: 44px; height: 44px; background: linear-gradient(135deg, #1e40af, #3b82f6); border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 18px; }
.header-info { display: flex; flex-direction: column; }
.report-title { font-size: 16px; font-weight: 600; color: #1e293b; margin: 0; letter-spacing: 0.5px; }
.report-subtitle { font-size: 11px; color: #94a3b8; margin: 2px 0 0 0; text-transform: uppercase; letter-spacing: 1px; }
.header-right { display: flex; gap: 24px; }
.meta-item { display: flex; flex-direction: column; align-items: flex-end; }
.meta-label { font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
.meta-value { font-size: 13px; color: #475569; font-family: 'SF Mono', monospace; margin-top: 2px; }
</style>
