<script setup lang="ts">
const navItems = [
  { name: '检测', path: '/' },
  { name: '会话检测', path: '/session' },
  { name: '插件测试', path: '/plugin' },
  { name: '个人', path: '/profile' }
];

</script>

<template>
  <nav class="top-navbar">
    <div class="nav-content">
      <div class="nav-brand">
        <i class="fa-solid fa-shapes"></i> LaRE System
      </div>
      <ul class="nav-links">
        <li v-for="item in navItems" :key="item.path">
          <router-link :to="item.path" active-class="active">
            {{ item.name }}
          </router-link>
        </li>
      </ul>
    </div>
  </nav>
</template>

<style scoped>
.top-navbar {
    width: 100%;
    height: 60px;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    position: fixed;
    top: 0;
    left: 0;
    z-index: 1000;
    display: flex;
    justify-content: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.nav-content {
    width: 100%;
    max-width: 1200px;
    padding: 0 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.nav-brand {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.nav-brand i {
    color: #60a5fa;
}

.nav-links {
    list-style: none;
    display: flex;
    gap: 30px;
    margin: 0;
    padding: 0;
}

.nav-links li a {
    color: #94a3b8;
    text-decoration: none;
    font-size: 0.95rem;
    font-weight: 500;
    transition: all 0.2s ease;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: 6px;
}

.nav-links li a:hover {
    color: #f1f5f9;
    background: rgba(255, 255, 255, 0.05);
}

.nav-links li a.active {
    color: #60a5fa;
    background: rgba(96, 165, 250, 0.1);
    font-weight: 600;
}
</style>
