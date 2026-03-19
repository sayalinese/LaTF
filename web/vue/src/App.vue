<script setup lang="ts">
import NavigationBar from './components/common/NavigationBar.vue';
</script>

<template>
  <NavigationBar />
  
  <div class="page-container">
    <router-view v-slot="{ Component, route }">
      <transition name="fade">
        <component :is="Component" :key="route.path" />
      </transition>
    </router-view>
  </div>
</template>

<style>
/* 页面切换动画：两页同时游走，无空白山谷 */
.page-container {
  position: relative;
  min-height: 100vh;
  width: 100%;
  background: var(--bg-gradient);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.18s ease;
  position: absolute;
  top: 0; left: 0; right: 0;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
