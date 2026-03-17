<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  config: any,
  loading: boolean,
  error: string | null
}>()

const emit = defineEmits<{
  (e: 'refresh'): void
}>()

const statusIcon = computed(() => {
  if (props.loading) return 'fa-solid fa-spinner fa-spin'
  if (props.error) return 'fa-solid fa-triangle-exclamation text-warn'
  if (props.config) return 'fa-solid fa-circle-check text-ok'
  return 'fa-solid fa-circle-question'
})

const copyText = () => {
    if(props.config?.ckpt_path) {
        navigator.clipboard.writeText(props.config.ckpt_path).then(()=>alert("Copied!"))
    }
}
</script>

<template>
  <div class="model-info-badge" :class="{'warn': error, 'ok': config && !error}">
    <i :class="statusIcon"></i>
    <span class="model-info-text">
        {{ loading ? '模型：加载中…' : (error ? `配置错误: ${error}` : `模型：${config?.model_version}`) }}
    </span>
    <button class="model-info-btn" @click="$emit('refresh')" title="刷新">
        <i class="fa-solid fa-rotate"></i>
    </button>
    <button class="model-info-btn" v-if="config?.ckpt_path" @click="copyText" title="复制路径">
        <i class="fa-solid fa-copy"></i>
    </button>
  </div>
</template>