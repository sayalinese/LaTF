<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { login, register, logout as apiLogout, fetchCurrentUser, updateProfile } from '../api';
import { useUserStore } from '../stores/user';

const { state: userState, loginUser, logoutUser, updateUser } = useUserStore();

const isLoggedIn = computed(() => userState.isLoggedIn);
const currentTab = ref<'login' | 'register'>('login');

// 表单数据
const loginForm = ref({ username: '', password: '' });
const registerForm = ref({ username: '', nickname: '', password: '', confirm: '' });

// 用户信息 — 直接读取 store
const userInfo = computed(() => userState);
// 判断头像是否为图片URL
const isAvatarUrl = computed(() => userState.avatar && userState.avatar.startsWith('/'));

const isEditing = ref(false);
const editForm = ref({ nickname: '', email: '' });
const avatarInput = ref<HTMLInputElement | null>(null);
const pendingAvatarFile = ref<File | null>(null);
const avatarPreview = ref('');

// 上传状态
const avatarUploading = ref(false);
const avatarError = ref('');

// 页面加载时始终从服务器同步最新用户状态（保证头像 URL 与数据库一致）
onMounted(async () => {
  try {
    const res = await fetchCurrentUser();
    if (res.success) {
      loginUser(res.user);
    }
  } catch { /* 未登录，忽略 */ }
});

// 行为逻辑
const handleLogin = async () => {
  if (loginForm.value.username && loginForm.value.password) {
    try {
      const res = await login({ username: loginForm.value.username, password: loginForm.value.password });
      if (res.success) {
        loginUser(res.user);
      } else {
        alert(res.message);
      }
    } catch (e: any) {
      alert("登录失败：" + (e.response?.data?.message || "网络错误"));
    }
  } else {
    alert("请输入用户名和密码！");
  }
};

const handleRegister = async () => {
  if (registerForm.value.password !== registerForm.value.confirm) {
    alert("两次密码不一致！");
    return;
  }
  if (registerForm.value.username && registerForm.value.password) {
    try {
      const res = await register({
        username: registerForm.value.username,
        password: registerForm.value.password,
        nickname: registerForm.value.nickname
      });
      if (res.success) {
        alert("注册成功！请登录。");
        currentTab.value = 'login';
        loginForm.value.username = registerForm.value.username;
      } else {
        alert(res.message);
      }
    } catch (e: any) {
      alert("注册失败：" + (e.response?.data?.message || "网络错误"));
    }
  } else {
    alert("请填写完整信息！");
  }
};

const handleLogout = async () => {
  try { await apiLogout(); } catch { /* ignore */ }
  logoutUser();
  loginForm.value.password = '';
};

const triggerAvatarUpload = () => {
  avatarInput.value?.click();
};

const handleAvatarChange = (event: Event) => {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file) return;
  pendingAvatarFile.value = file;
  avatarPreview.value = URL.createObjectURL(file);
  // 立即上传头像
  uploadAvatar(file);
  target.value = '';
};

const uploadAvatar = async (file: File) => {
  avatarUploading.value = true;
  avatarError.value = '';
  try {
    const res = await updateProfile({ avatar: file });
    if (res.success) {
      updateUser({ avatar: res.user.avatar, nickname: res.user.nickname });
      avatarPreview.value = '';
      pendingAvatarFile.value = null;
    } else {
      avatarPreview.value = '';
      pendingAvatarFile.value = null;
      avatarError.value = res.message || '头像上传失败';
    }
  } catch (e: any) {
    console.error('头像上传失败:', e);
    avatarPreview.value = '';
    pendingAvatarFile.value = null;
    avatarError.value = e.response?.data?.message || '头像上传失败，请重试';
  } finally {
    avatarUploading.value = false;
  }
};

const saveProfile = async () => {
  try {
    const payload: { nickname?: string; avatar?: File } = {};
    if (editForm.value.nickname && editForm.value.nickname !== userState.nickname) {
      payload.nickname = editForm.value.nickname;
    }
    if (pendingAvatarFile.value) {
      payload.avatar = pendingAvatarFile.value;
    }
    if (Object.keys(payload).length > 0) {
      const res = await updateProfile(payload);
      if (res.success) {
        updateUser({ avatar: res.user.avatar, nickname: res.user.nickname });
      }
    }
  } catch (e: any) {
    alert('保存失败: ' + (e.response?.data?.message || '网络错误'));
  }
  isEditing.value = false;
  pendingAvatarFile.value = null;
  avatarPreview.value = '';
};

const cancelEdit = () => {
  editForm.value = { nickname: userState.nickname, email: userState.email };
  isEditing.value = false;
  pendingAvatarFile.value = null;
  avatarPreview.value = '';
};
</script>

<template>
  <div class="profile-layout">
    
    <!-- ================= 未登录状态：炫酷双窗格卡片 ================= -->
    <transition name="fade-scale" mode="out-in">
      <div v-if="!isLoggedIn" class="auth-showcase">
        <div class="auth-card-modern">
          
          <!-- 左侧：视觉展示区 -->
          <div class="auth-visual">
            <div class="visual-content">
              <div class="logo-box">
                <i class="fa-solid fa-shield-halved"></i>
              </div>
              <h2>LaFT System</h2>
              <p>多模态 · 全频域<br>AIGC 图像伪造检测平台</p>
            </div>
            
            <!-- 背景动效元素 -->
            <div class="floating-orb orb-1"></div>
            <div class="floating-orb orb-2"></div>
            <div class="floating-orb orb-3"></div>
            <div class="glass-overlay"></div>
          </div>

          <!-- 右侧：表单区 -->
          <div class="auth-form-section">
            <transition name="slide-up" mode="out-in">
              <div :key="currentTab">
                <!-- 登录表单 -->
                <div v-if="currentTab === 'login'" class="form-panel">
                  <div class="form-header">
                    <h3>欢迎回来</h3>
                    <p>登录检测平台系统账号</p>
                  </div>
                  
                  <div class="input-group">
                    <div class="input-icon-wrapper">
                      <i class="fa-solid fa-user"></i>
                      <input type="text" v-model="loginForm.username" placeholder="邮箱 / 用户名" />
                    </div>
                  </div>
                  
                  <div class="input-group">
                    <div class="input-icon-wrapper">
                      <i class="fa-solid fa-lock"></i>
                      <input type="password" v-model="loginForm.password" placeholder="密码" @keyup.enter="handleLogin" />
                    </div>
                  </div>
                  
                  <div class="form-options">
                    <label class="remember-me">
                      <input type="checkbox"/> 
                      <span class="custom-checkbox"></span>
                      保持登录状态
                    </label>
                    <a href="#" class="forgot-pwd">忘记密码?</a>
                  </div>
                  
                  <button class="cyber-btn" @click="handleLogin">
                    <span>安全登录 <i class="fa-solid fa-arrow-right"></i></span>
                  </button>
                  
                  <div class="toggle-text">
                    尚未加入平台？ <a href="#" @click.prevent="currentTab = 'register'">立即创建账号</a>
                  </div>
                </div>

                <!-- 注册表单 -->
                <div v-if="currentTab === 'register'" class="form-panel register-panel">
                  <div class="form-header">
                    <h3>加入 LaFT</h3>
                    <p>探索多模态智能图像鉴定</p>
                  </div>
                  
                  <div class="input-group">
                    <div class="input-icon-wrapper">
                      <i class="fa-solid fa-user"></i>
                      <input type="text" v-model="registerForm.username" placeholder="设置用户名" />
                    </div>
                  </div>
                  
                  <div class="input-group">
                    <div class="input-icon-wrapper">
                      <i class="fa-solid fa-id-card"></i>
                      <input type="text" v-model="registerForm.nickname" placeholder="显示的鉴核昵称" />
                    </div>
                  </div>
                  
                  <div class="input-group">
                    <div class="input-icon-wrapper">
                      <i class="fa-solid fa-lock"></i>
                      <input type="password" v-model="registerForm.password" placeholder="设置高强度密码" />
                    </div>
                  </div>
                  
                  <div class="input-group">
                    <div class="input-icon-wrapper">
                      <i class="fa-solid fa-check-double"></i>
                      <input type="password" v-model="registerForm.confirm" placeholder="再次确认密码" />
                    </div>
                  </div>
                  
                  <button class="cyber-btn register-variant" @click="handleRegister">
                    <span>创建账号 <i class="fa-solid fa-user-plus"></i></span>
                  </button>
                  
                  <div class="toggle-text">
                    已有账户身份？ <a href="#" @click.prevent="currentTab = 'login'">返回凭证登录</a>
                  </div>
                </div>
              </div>
            </transition>
          </div>
        </div>
      </div>
    </transition>

    <!-- ================= 已登录状态：个人信息面板 ================= -->
    <transition name="fade-scale" mode="out-in">
      <div v-if="isLoggedIn" class="profile-container">
        <div class="profile-card glass-panel">
          
          <div class="profile-header">
            <input type="file" ref="avatarInput" accept="image/*" style="display:none" @change="handleAvatarChange" />
            <div class="avatar-large clickable" @click="triggerAvatarUpload" :title="avatarUploading ? '上传中...' : '点击更换头像'">
              <img v-if="avatarPreview" :src="avatarPreview" class="avatar-img" alt="头像" />
              <img v-else-if="isAvatarUrl" :src="userInfo.avatar" class="avatar-img" alt="头像" />
              <span v-else>{{ userInfo.avatar }}</span>
              <div class="avatar-overlay">
                <i v-if="avatarUploading" class="fa-solid fa-spinner fa-spin"></i>
                <i v-else class="fa-solid fa-camera"></i>
              </div>
            </div>
            <div v-if="avatarError" class="avatar-error-tip">
              <i class="fa-solid fa-circle-exclamation"></i> {{ avatarError }}
            </div>
            <div class="user-titles">
              <h2>{{ userInfo.nickname }}</h2>
            </div>
            <button class="logout-btn" @click="handleLogout" title="退出登录">
              <i class="fa-solid fa-power-off"></i>
            </button>
          </div>

          <div class="profile-content">
            <div class="section-title">
              <h3><i class="fa-solid fa-address-card"></i> 账号身份卡</h3>
              <button v-if="!isEditing" class="edit-btn" @click="isEditing = true">
                <i class="fa-solid fa-pen-to-square"></i> 编辑资料
              </button>
            </div>

            <div class="info-grid">
              <div class="info-item">
                <label>系统认证ID</label>
                <div class="info-value disabled">{{ userInfo.id }}</div>
              </div>

              <div class="info-item">
                <label>登录身份</label>
                <div class="info-value disabled">{{ userInfo.username }}</div>
              </div>

              <div class="info-item">
                <label>展示昵称</label>
                <div v-if="!isEditing" class="info-value">{{ userInfo.nickname }}</div>
                <input v-else type="text" class="edit-input" v-model="editForm.nickname" />
              </div>

              <div class="info-item">
                <label>安全绑定邮箱</label>
                <div v-if="!isEditing" class="info-value">{{ userInfo.email || '未绑定安全邮箱' }}</div>
                <input v-else type="email" class="edit-input" v-model="editForm.email" />
              </div>
            </div>

            <transition name="slide-up">
              <div v-if="isEditing" class="edit-actions">
                <button class="secondary-btn" @click="cancelEdit">撤销更改</button>
                <button class="cyber-btn save-btn" @click="saveProfile">
                  <span>保存档案 <i class="fa-solid fa-cloud-arrow-up"></i></span>
                </button>
              </div>
            </transition>
          </div>

          <!-- 用户统计信息 -->
          <div class="stats-section">
            <div class="stat-box">
              <div class="stat-icon"><i class="fa-solid fa-microscope"></i></div>
              <div class="stat-info">
                <div class="stat-num">{{ userInfo.historyCount }}</div>
                <div class="stat-label">历史检测鉴伪</div>
              </div>
            </div>
            <div class="stat-box">
              <div class="stat-icon"><i class="fa-solid fa-scale-balanced"></i></div>
              <div class="stat-info">
                <div class="stat-num">{{ userInfo.disputesCount }}</div>
                <div class="stat-label">发起的纠纷法庭</div>
              </div>
            </div>
            <div class="stat-box highlight">
              <div class="stat-icon"><i class="fa-solid fa-bolt"></i></div>
              <div class="stat-info">
                <div class="stat-num">{{ userInfo.accuracy }}</div>
                <div class="stat-label">AI 综合准确率</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </transition>
    
  </div>
</template>

<style scoped>
.profile-layout {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 80px);
  width: 100vw;
  margin-top: 60px; /* Adjust to match navbar offset */
  padding: 20px;
  background: var(--bg-gradient);
  position: relative;
  overflow: hidden;
}

/* 页面级别发光特效背景 */
.profile-layout::before {
  content: '';
  position: absolute;
  top: -20%;
  left: -10%;
  width: 50%;
  height: 50%;
  background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
  filter: blur(80px);
  z-index: 0;
  pointer-events: none;
}
.profile-layout::after {
  content: '';
  position: absolute;
  bottom: -20%;
  right: -10%;
  width: 50%;
  height: 50%;
  background: radial-gradient(circle, rgba(168, 85, 247, 0.15) 0%, transparent 60%);
  filter: blur(80px);
  z-index: 0;
  pointer-events: none;
}

/* 确保所有内容的层级在背景特效之上 */
.auth-showcase, .profile-container {
  position: relative;
  z-index: 10;
}

/* ================= 炫酷双窗格卡片 ================= */
.auth-showcase {
  width: 100%;
  max-width: 1000px;
  perspective: 1000px;
}
.auth-card-modern {
  display: flex;
  background: var(--panel-bg);
  backdrop-filter: blur(20px);
  border-radius: 24px;
  border: 1px solid var(--border-color-light);
  box-shadow: 0 30px 60px -15px rgba(0, 0, 0, 0.6), inset 0 1px 0 var(--border-color);
  overflow: hidden;
  height: auto;
  min-height: 500px;
}

/* 左侧背景 */
.auth-visual {
  flex: 1;
  position: relative;
  background: linear-gradient(135deg, var(--card-bg-solid) 0%, var(--panel-bg-solid) 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.visual-content {
  position: relative;
  z-index: 10;
  text-align: center;
  padding: 40px;
}
.logo-box {
  width: 80px; height: 80px;
  margin: 0 auto 20px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
  border: 1px solid rgba(168, 85, 247, 0.4);
  border-radius: 20px;
  display: flex; align-items: center; justify-content: center;
  font-size: 2.5rem; color: #c084fc;
  box-shadow: 0 10px 25px rgba(168, 85, 247, 0.3);
  backdrop-filter: blur(10px);
}
.visual-content h2 {
  font-size: 2.2rem;
  font-weight: 700;
  background: linear-gradient(to right, #e2e8f0, var(--text-muted));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 12px;
  letter-spacing: 2px;
  white-space: nowrap;
}
.visual-content p {
  color: var(--text-muted); line-height: 1.6; font-size: 1rem;
}

/* 动效光球 */
.floating-orb {
  position: absolute; border-radius: 50%; filter: blur(40px); opacity: 0.5;
  animation: float 10s infinite ease-in-out alternate;
}
.orb-1 { width: 300px; height: 300px; background: #4f46e5; top: -100px; left: -100px; animation-delay: 0s; }
.orb-2 { width: 250px; height: 250px; background: #9333ea; bottom: -50px; right: -50px; animation-delay: -3s; }
.orb-3 { width: 200px; height: 200px; background: #2563eb; top: 40%; left: 30%; animation-delay: -6s; opacity: 0.3; }

@keyframes float {
  0% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(30px, 30px) scale(1.1); }
  100% { transform: translate(-20px, 20px) scale(0.9); }
}
.glass-overlay {
  position: absolute; inset: 0; background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMiIgY3k9IjIiIHI9IjEiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wNykiLz48L3N2Zz4=');
  z-index: 5; mix-blend-mode: overlay;
}

/* 右侧表单区 */
.auth-form-section {
  flex: 1;
  background: var(--panel-bg);
  padding: 40px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  position: relative;
}
.form-panel {
  width: 100%; max-width: 320px; margin: 0 auto;
}
.form-header { margin-bottom: 30px; text-align: left; }
.form-header h3 { font-size: 1.6rem; color: var(--text-main); margin-bottom: 6px; }
.form-header p { color: #64748b; font-size: 0.9rem; }

.input-group { margin-bottom: 16px; }
.input-icon-wrapper {
  position: relative; display: flex; align-items: center;
}
.input-icon-wrapper i {
  position: absolute; left: 16px; color: #64748b; transition: 0.3s;
}
.input-icon-wrapper input {
  width: 100%; height: 48px;
  background: var(--border-color-light);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 0 16px 0 46px;
  color: #fff; font-size: 0.95rem;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.input-icon-wrapper input:focus {
  background: var(--border-color-light);
  border-color: var(--primary-color);
  box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
  outline: none;
}
.input-icon-wrapper input:focus + i, .input-icon-wrapper input:not(:placeholder-shown) ~ i {
  color: #8b5cf6;
}

.form-options {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 24px; font-size: 0.85rem;
}
.remember-me {
  display: flex; align-items: center; gap: 8px; color: var(--text-muted); cursor: pointer;
}
.remember-me input { display: none; }
.custom-checkbox {
  width: 16px; height: 16px; border: 1px solid var(--border-color); border-radius: 4px;
  background: rgba(0,0,0,0.2); display: flex; align-items: center; justify-content: center;
  transition: 0.2s;
}
.remember-me input:checked + .custom-checkbox {
  background: var(--primary-color); border-color: var(--primary-color);
}
.remember-me input:checked + .custom-checkbox::after {
  content: '\2714'; font-family: 'Font Awesome 6 Free'; font-weight: 900; color: white; font-size: 10px;
}
.forgot-pwd { color: #8b5cf6; text-decoration: none; transition: 0.2s; }
.forgot-pwd:hover { color: #a855f7; text-decoration: underline; }

.cyber-btn {
  width: 100%; height: 48px;
  background: linear-gradient(135deg, #4f46e5, #7e22ce);
  color: white; border: none; border-radius: 12px;
  font-size: 1rem; font-weight: 600; letter-spacing: 1px;
  cursor: pointer; position: relative; overflow: hidden;
  transition: all 0.3s;
  box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
}
.cyber-btn span { position: relative; z-index: 2; display: flex; align-items: center; justify-content: center; gap: 8px;}
.cyber-btn::before {
  content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
  background: linear-gradient(90deg, transparent, var(--border-color), transparent);
  transition: 0.5s;
}
.cyber-btn:hover { transform: translateY(-2px); box-shadow: 0 12px 25px rgba(99, 102, 241, 0.4); }
.cyber-btn:hover::before { left: 100%; }

.register-variant { background: linear-gradient(135deg, #0ea5e9, #3b82f6); box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3);}
.register-variant:hover { box-shadow: 0 12px 25px rgba(14, 165, 233, 0.4);}

.toggle-text {
  text-align: center; margin-top: 24px; font-size: 0.9rem; color: #64748b;
}
.toggle-text a { color: #60a5fa; font-weight: 500; text-decoration: none; padding-bottom: 2px; border-bottom: 1px dashed transparent; transition: 0.2s;}
.toggle-text a:hover { color: #93c5fd; border-color: #93c5fd; }


/* ================= 个人信息面板优化 ================= */
.profile-container { width: 100%; max-width: 850px; }
.glass-panel {
  background: var(--panel-bg); backdrop-filter: blur(24px);
  border: 1px solid var(--border-color-light); border-radius: 24px;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5); padding: 40px;
}
.profile-header {
  border-bottom: 1px solid var(--border-color-light); padding-bottom: 30px; margin-bottom: 30px;
  display: flex; align-items: center; justify-content: space-between;
}
.avatar-large {
  width: 80px; height: 80px; border-radius: 20px;
  background: linear-gradient(135deg, var(--primary-color), #c084fc);
  color: white; font-size: 2.5rem; font-weight: bold;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
  position: relative; overflow: hidden;
}
.avatar-large.clickable { cursor: pointer; }
.avatar-large .avatar-img {
  width: 100%; height: 100%; object-fit: cover; border-radius: 20px;
}
.avatar-overlay {
  position: absolute; inset: 0;
  background: rgba(0,0,0,0.45);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem; color: #fff;
  opacity: 0; transition: opacity 0.25s;
}
.avatar-large.clickable:hover .avatar-overlay { opacity: 1; }
.avatar-error-tip {
  font-size: 0.8rem; color: #f87171;
  display: flex; align-items: center; gap: 5px;
  margin-top: 8px; text-align: center;
}
.user-titles {
  flex: 1; margin-left: 24px;
}
.user-titles h2 {
  font-size: 1.8rem; color: var(--text-main); margin-bottom: 8px;
}
.section-title {
  display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;
}
.section-title h3 {
  font-size: 1.2rem; color: var(--text-main); display: flex; align-items: center; gap: 8px;
}
.section-title h3 i {
  color: var(--primary-color);
}
.edit-btn {
  background: var(--border-color-light); border: 1px solid var(--border-color);
  color: var(--text-main); padding: 8px 16px; border-radius: 8px; font-size: 0.9rem;
  cursor: pointer; transition: 0.2s; display: flex; align-items: center; gap: 6px;
}
.edit-btn:hover { background: var(--primary-color); color: white; border-color: var(--primary-color); }
.info-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;
}
.info-item {
  background: var(--border-color-light); border: 1px solid var(--border-color-light);
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); transition: 0.3s;
  padding: 16px 20px; border-radius: 12px;
}
.info-item label {
  display: block; color: var(--text-muted); font-size: 0.85rem; margin-bottom: 8px;
}
.info-item .info-value {
  color: var(--text-main); font-size: 1.05rem; font-weight: 500;
}
.info-item .info-value.disabled {
  color: var(--text-muted); cursor: not-allowed;
}
.edit-input {
  width: 100%; padding: 8px 12px;
  background: var(--card-bg); border: 1px solid var(--primary-color);
  border-radius: 8px; color: var(--text-main); font-size: 1rem;
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.1); outline: none;
}
.edit-actions {
  display: flex; justify-content: flex-end; gap: 16px; margin-top: 30px;
}
.secondary-btn {
  background: transparent; border: 1px solid var(--border-color);
  color: var(--text-muted); padding: 10px 24px; border-radius: 10px; cursor: pointer; transition: 0.2s;
}
.secondary-btn:hover { background: var(--border-color-light); color: var(--text-main); }
.role-badge {
  background: linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
  border: 1px solid rgba(16, 185, 129, 0.2); box-shadow: 0 4px 10px rgba(16, 185, 129, 0.1);
  color: var(--real-color); padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500;
  display: inline-flex; align-items: center; gap: 6px;
}
.logout-btn {
  background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2);
  color: var(--fake-color); border-radius: 12px; transition: all 0.3s;
  width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem; cursor: pointer; margin-left: auto;
}
.logout-btn:hover { background: #ef4444; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3); color: white;}

.info-item {
  background: var(--border-color-light); border: 1px solid var(--border-color-light);
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); transition: 0.3s;
  padding: 16px 20px; border-radius: 12px;
}
.info-item:hover { background: var(--border-color-light); border-color: var(--primary-color);}

.stats-section { margin-top: 40px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; }
.stat-box {
  display: flex; align-items: center; gap: 16px; text-align: left;
  background: var(--card-bg); border: 1px solid var(--border-color-light); padding: 24px;
  transition: 0.3s; position: relative; overflow: hidden;
}
.stat-box:hover { transform: translateY(-3px); background: var(--card-bg); border-color: var(--border-color); }
.stat-icon {
  width: 48px; height: 48px; border-radius: 12px; background: var(--border-color-light);
  display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: #a8bdff;
}
.highlight .stat-icon { background: linear-gradient(135deg, #f59e0b, #d97706); color: white;}
.stat-num { font-size: 1.8rem; letter-spacing: 1px;}
.stat-label { color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }

/* 动画类 */
.slide-up-enter-active, .slide-up-leave-active, .fade-scale-enter-active, .fade-scale-leave-active {
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}
.slide-up-enter-from, .slide-up-leave-to { opacity: 0; transform: translateY(20px); }
.fade-scale-enter-from, .fade-scale-leave-to { opacity: 0; transform: scale(0.95); }

@media (max-width: 800px) {
  .auth-card-modern { flex-direction: column; height: auto; }
  .auth-visual { padding: 40px 20px; flex: none; }
  .auth-form-section { padding: 30px 20px; }
}
</style>
