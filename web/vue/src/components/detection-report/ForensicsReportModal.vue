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

const isSavingImage = ref(false);
const scrollContainerRef = ref<HTMLElement | null>(null);
const exportVersion = 'IMG_EXPORT_V3';

const handleExportImage = async () => {
  const el = scrollContainerRef.value?.querySelector('.report-document') as HTMLElement | null;
  if (!el) return;

  isSavingImage.value = true;
  try {
    const html2canvas = (await import('html2canvas')).default;
    const canvas = await html2canvas(el, {
      scale: 2,
      useCORS: true,
      logging: false,
      backgroundColor: '#ffffff',
      onclone: (doc) => {
        const report = doc.getElementById('report-document-content');
        if (!report) return;
        const wm = report.querySelector('.watermark') as HTMLElement | null;
        if (wm) wm.remove();

        // 强制移除所有可能影响颜色的 CSS 类和内联样式，并直接赋予最纯粹的样式
        const logo = report.querySelector('.logo') as HTMLElement | null;
        const logoIcon = report.querySelector('.logo i') as HTMLElement | null;
        const logoText = report.querySelector('.logo span') as HTMLElement | null;
        
        if (logo) {
          logo.removeAttribute('style');
          logo.setAttribute('style', 'color: #000000 !important; background: none !important; opacity: 1 !important; filter: none !important; mix-blend-mode: normal !important;');
        }
        if (logoIcon) {
          logoIcon.removeAttribute('style');
          logoIcon.setAttribute('style', 'color: #000000 !important; background: none !important; opacity: 1 !important;');
        }
        if (logoText) {
          logoText.removeAttribute('style');
          logoText.setAttribute('style', 'color: #000000 !important; background: none !important; opacity: 1 !important;');
        }

        const reportId = report.querySelector('.report-id') as HTMLElement | null;
        const reportTitle = report.querySelector('.report-title') as HTMLElement | null;
        const headerMeta = report.querySelector('.header-meta') as HTMLElement | null;
        const reportHeader = report.querySelector('.report-header') as HTMLElement | null;
        
        if (reportId) {
          reportId.style.setProperty('color', '#000000', 'important');
          reportId.style.setProperty('opacity', '1', 'important');
          reportId.style.setProperty('filter', 'none', 'important');
        }
        if (reportTitle) {
          reportTitle.style.setProperty('color', '#000000', 'important');
          reportTitle.style.setProperty('font-weight', '800', 'important');
          reportTitle.style.setProperty('opacity', '1', 'important');
          reportTitle.style.setProperty('filter', 'none', 'important');
        }
        if (headerMeta) {
          headerMeta.style.setProperty('color', '#000000', 'important');
          headerMeta.style.setProperty('opacity', '1', 'important');
          headerMeta.style.setProperty('filter', 'none', 'important');
        }
        if (reportHeader) {
          reportHeader.style.setProperty('border-bottom', '2px solid #000000', 'important');
          reportHeader.style.setProperty('opacity', '1', 'important');
          reportHeader.style.setProperty('filter', 'none', 'important');
          reportHeader.style.setProperty('mix-blend-mode', 'normal', 'important');
          reportHeader.style.setProperty('background', '#ffffff', 'important');
        }
      }
    });
    
    const timestamp = new Date().toISOString().slice(0, 10);
    const link = document.createElement('a');
    link.download = `潜迹寻真-取证报告-${exportVersion}-${timestamp}.png`;
    link.href = canvas.toDataURL('image/png', 1.0);
    link.click();
  } catch (error) {
    console.error('Failed to export image:', error);
  } finally {
    isSavingImage.value = false;
  }
};
</script>

<template>
  <div class="forensics-report-overlay" @click.self="emit('close')">
    <div class="modal-wrapper">
      
      <!-- Toolbar -->
      <div class="toolbar no-print">
        <div class="toolbar-left">
          <i class="fa-solid fa-file-shield"></i> 取证鉴定报告预览（{{ exportVersion }}）
        </div>
        <div class="toolbar-right">
          <button class="btn-print" @click="handleExportImage" :disabled="isSavingImage">
            <i :class="isSavingImage ? 'fa-solid fa-spinner fa-spin' : 'fa-solid fa-image'"></i>
            {{ isSavingImage ? '生成中...' : '保存为长图 V3' }}
          </button>
          <button class="btn-close" @click="emit('close')">
            <i class="fa-solid fa-xmark"></i>
          </button>
        </div>
      </div>

      <div class="document-scroll-container" ref="scrollContainerRef">
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
