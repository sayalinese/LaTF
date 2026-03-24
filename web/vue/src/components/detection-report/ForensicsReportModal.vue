<script setup lang="ts">
import { ref } from 'vue';
import type { ForensicsReportData } from './types';
import ReportDocument from './ReportDocument.vue';

const props = defineProps<{
  reportData: ForensicsReportData;
}>();

const emit = defineEmits<{
  (e: 'close'): void;
}>();

const isSavingPdf = ref(false);

const handleExportPdf = async () => {
  const el = document.querySelector('.report-document') as HTMLElement;
  if (!el) return;

  isSavingPdf.value = true;
  try {
    const html2pdf = (await import('html2pdf.js')).default;
    const timestamp = new Date().toISOString().slice(0, 10);
    await html2pdf()
      .set({
        margin: 0,
        filename: `潜迹寻真-取证报告-${timestamp}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, logging: false },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
      })
      .from(el)
      .save();
  } finally {
    isSavingPdf.value = false;
  }
};
</script>

<template>
  <div class="forensics-report-overlay" @click.self="emit('close')">
    <div class="modal-wrapper">
      
      <!-- Toolbar -->
      <div class="toolbar no-print">
        <div class="toolbar-left">
          <i class="fa-solid fa-file-shield"></i> 取证鉴定报告预览
        </div>
        <div class="toolbar-right">
          <button class="btn-print" @click="handleExportPdf" :disabled="isSavingPdf">
            <i :class="isSavingPdf ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-file-pdf'"></i>
            {{ isSavingPdf ? '生成中...' : '保存 PDF' }}
          </button>
          <button class="btn-close" @click="emit('close')">
            <i class="fa-solid fa-xmark"></i>
          </button>
        </div>
      </div>

      <!-- Scrollable Content -->
      <div class="document-scroll-container">
        <ReportDocument :data="reportData" />
      </div>

    </div>
  </div>
</template>

<style scoped>
.forensics-report-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.85); /* Slate 900 heavy */
  backdrop-filter: blur(4px);
  z-index: 9999;
  display: flex;
  justify-content: center;
  align-items: center;
}

.modal-wrapper {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.toolbar {
  height: 60px;
  background: #1e293b;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 30px;
  color: white;
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  z-index: 10;
}

.toolbar-left {
  font-size: 18px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
}

.toolbar-right {
  display: flex;
  gap: 15px;
}

button {
  border: none;
  border-radius: 6px;
  padding: 8px 16px;
  font-size: 14px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s;
}

.btn-print {
  background: #333333;
  color: white;
}
.btn-print:hover { background: #111111; }

.btn-close {
  background: transparent;
  color: #94a3b8;
  font-size: 20px;
  padding: 8px;
}
.btn-close:hover { color: white; background: rgba(255,255,255,0.1); }

.document-scroll-container {
  flex: 1;
  overflow-y: auto;
  padding: 40px 20px;
  /* subtle pattern background for the wrapper */
  background-color: #e2e8f0;
  background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
  background-size: 20px 20px;
}

/* Print Overrides */
@media print {
  .no-print {
    display: none !important;
  }
  .forensics-report-overlay {
    position: static;
    background: transparent;
    display: block;
  }
  .modal-wrapper {
    height: auto;
    display: block;
  }
  .document-scroll-container {
    overflow: visible;
    padding: 0;
    background: white;
  }
}
</style>
<style>
/* Global print styles to hide everything except the modal */
@media print {
  body > * {
    display: none !important;
  }
  body > .forensics-report-overlay {
    display: block !important;
  }
}
</style>