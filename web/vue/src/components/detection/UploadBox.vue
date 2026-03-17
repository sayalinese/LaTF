<script setup lang="ts">
import { ref } from 'vue'

const emit = defineEmits<{
  (e: 'file-selected', file: File): void
}>()

const isDragging = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)

const handleDrop = (e: DragEvent) => {
  isDragging.value = false
  if (e.dataTransfer?.files.length) {
    const file = e.dataTransfer.files[0]
    if (file.type.startsWith('image/')) {
      emit('file-selected', file)
    }
  }
}

const handleFileSelect = (e: Event) => {
  const target = e.target as HTMLInputElement
  if (target.files?.length) {
    emit('file-selected', target.files[0])
    // Reset input so the same file can be selected again
    target.value = ''
  }
}

const triggerSelect = () => {
  fileInput.value?.click()
}
</script>

<template>
  <div 
    class="upload-section" 
    :class="{ 'drag-active': isDragging }"
    @dragover.prevent="isDragging = true"
    @dragleave.prevent="isDragging = false"
    @drop.prevent="handleDrop"
    @click="triggerSelect"
  >
    <input 
      type="file" 
      ref="fileInput" 
      accept="image/*" 
      hidden 
      @change="handleFileSelect"
    >
    <div class="upload-content">
      <div class="icon-circle">
        <i class="fa-solid fa-cloud-arrow-up"></i>
      </div>
      <h3>上传图片</h3>
      <p>拖拽或点击浏览</p>
      <span class="file-support">支持 JPG, PNG, WEBP</span>
    </div>
  </div>
</template>
