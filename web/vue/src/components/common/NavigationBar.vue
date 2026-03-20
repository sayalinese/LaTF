<script setup lang="ts">
import { ref, watchEffect, computed } from 'vue';
import { useUserStore } from '../../stores/user';

const { state: userState } = useUserStore();
const isLoggedIn = computed(() => userState.isLoggedIn);

const navItems = [
  { name: '检测', path: '/' },
  { name: '会话检测', path: '/session' },
  { name: '插件测试', path: '/plugin' },
  { name: '个人', path: '/profile' }
];

const theme = ref(localStorage.getItem('theme') || 'dark');
const toggleTheme = () => {
    theme.value = theme.value === 'dark' ? 'light' : 'dark';
};

watchEffect(() => {
    document.documentElement.setAttribute('data-theme', theme.value);
    localStorage.setItem('theme', theme.value);
});

</script>

<template>
  <nav class="top-navbar">
    <div class="nav-content">
      <div class="nav-brand">
        <i class="fa-solid fa-shapes"></i> LaFT System
      </div>
      <ul class="nav-links">
        <li v-for="item in navItems" :key="item.path">
          <router-link :to="item.path" active-class="active">
            {{ item.name }}
          </router-link>
        </li>
      </ul>
      <div class="nav-right">
        <div class="theme-switch" @click="toggleTheme" title="切换主题">
            <i class="fa-solid" :class="theme === 'dark' ? 'fa-sun' : 'fa-moon'"></i>
        </div>
        <router-link to="/profile" class="user-indicator">
          <template v-if="isLoggedIn">
            <div class="user-avatar" :title="userState.nickname">
              <img v-if="userState.avatar && userState.avatar.startsWith('/')" :src="userState.avatar" class="avatar-img" alt="" />
              <span v-else>{{ userState.avatar || userState.nickname?.[0] || '?' }}</span>
            </div>
          </template>
          <template v-else>
            <div class="login-btn">登录</div>
          </template>
        </router-link>
      </div>
    </div>
  </nav>
</template>

<style scoped>
.nav-right {
    display: flex;
    align-items: center;
    gap: 16px;
}
.user-indicator {
    text-decoration: none;
}
.user-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--primary-color);
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.85rem;
    cursor: pointer;
    transition: box-shadow 0.2s;
    overflow: hidden;
}
.user-avatar .avatar-img {
    width: 100%; height: 100%; object-fit: cover; border-radius: 50%;
}
.user-avatar:hover {
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3);
}
.login-btn {
    padding: 6px 16px;
    border-radius: 6px;
    background: var(--primary-color);
    color: #fff;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.2s;
}
.login-btn:hover {
    opacity: 0.85;
}
.theme-switch {
    cursor: pointer;
    font-size: 1.2rem;
    color: var(--text-muted);
    transition: color 0.3s;
    display: flex;
    align-items: center;
}
.theme-switch:hover {
    color: var(--primary-color);
}
.top-navbar {
    width: 100%;
    height: 60px;
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border-color-light);
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
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 10px;
}

.nav-brand i {
    color: var(--primary-color);
}

.nav-links {
    list-style: none;
    display: flex;
    gap: 30px;
    margin: 0;
    padding: 0;
}

.nav-links li a {
    color: var(--text-muted);
    text-decoration: none;
    font-size: 0.95rem;
    font-weight: 500;
    transition: all 0.2s ease;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: 6px;
}

.nav-links li a:hover {
    color: var(--text-main);
    background: var(--border-color-light);
}

.nav-links li a.active {
    color: var(--primary-color);
    background: rgba(99, 102, 241, 0.1);
    font-weight: 600;
}
</style>
