<script setup lang="ts">
import { ref } from 'vue';

// 模拟状态管理
const isLoggedIn = ref(false);
const currentTab = ref<'login' | 'register'>('login'); // 用于在未登录时切换界面

// 表单数据
const loginForm = ref({ username: '', password: '' });
const registerForm = ref({ username: '', nickname: '', password: '', confirm: '' });

// 用户信息 (模拟已登录数据)
const userInfo = ref({
  id: 'u_001',
  username: 'admin',
  nickname: '平台鉴定师',
  email: 'admin@lare.com',
  avatar: 'A'
});

const isEditing = ref(false);
const editForm = ref({ ...userInfo.value });

// 行为逻辑
const handleLogin = () => {
  if (loginForm.value.username && loginForm.value.password) {
    // 模拟后端校验成功
    userInfo.value.username = loginForm.value.username;
    userInfo.value.avatar = loginForm.value.username.charAt(0).toUpperCase();
    editForm.value = { ...userInfo.value };
    isLoggedIn.value = true;
  } else {
    alert("请输入用户名和密码！");
  }
};

const handleRegister = () => {
  if (registerForm.value.password !== registerForm.value.confirm) {
    alert("两次密码不一致！");
    return;
  }
  if (registerForm.value.username && registerForm.value.password) {
    alert("注册成功！请登录。");
    currentTab.value = 'login';
    loginForm.value.username = registerForm.value.username;
  }
};

const handleLogout = () => {
  isLoggedIn.value = false;
  loginForm.value.password = ''; // 清除密码
};

const saveProfile = () => {
  userInfo.value = { ...editForm.value };
  // 模拟保存至后端
  isEditing.value = false;
};

const cancelEdit = () => {
  editForm.value = { ...userInfo.value };
  isEditing.value = false;
};
</script>

<template>
  <div class="profile-layout">
    
    <!-- ================= 未登录状态：登录/注册卡片 ================= -->
    <div v-if="!isLoggedIn" class="auth-container">
      <div class="auth-card glass-panel">
        
        <!-- 登录表单 -->
        <template v-if="currentTab === 'login'">
          <div class="auth-header">
            <i class="fa-solid fa-user-lock"></i>
            <h2>系统登录</h2>
            <p>欢迎回到 LaRE AIGC 图像鉴定系统</p>
          </div>
          
          <div class="form-group">
            <label>用户名</label>
            <div class="input-wrapper">
              <i class="fa-solid fa-user"></i>
              <input type="text" v-model="loginForm.username" placeholder="请输入用户名" />
            </div>
          </div>
          <div class="form-group">
            <label>密码</label>
            <div class="input-wrapper">
              <i class="fa-solid fa-key"></i>
              <input type="password" v-model="loginForm.password" placeholder="请输入密码" @keyup.enter="handleLogin" />
            </div>
          </div>
          
          <button class="primary-btn" @click="handleLogin">登 录</button>
          
          <div class="auth-footer">
            没有账号？ <a href="#" @click.prevent="currentTab = 'register'">立即注册</a>
          </div>
        </template>

        <!-- 注册表单 -->
        <template v-if="currentTab === 'register'">
          <div class="auth-header">
            <i class="fa-solid fa-user-plus"></i>
            <h2>创建新账号</h2>
            <p>加入 LaRE，探索多模态鉴定</p>
          </div>
          
          <div class="form-group">
            <label>用户名</label>
            <div class="input-wrapper">
              <i class="fa-solid fa-user"></i>
              <input type="text" v-model="registerForm.username" placeholder="设置登录用户名" />
            </div>
          </div>
          <div class="form-group">
            <label>昵称</label>
            <div class="input-wrapper">
              <i class="fa-solid fa-id-card"></i>
              <input type="text" v-model="registerForm.nickname" placeholder="系统中显示的昵称" />
            </div>
          </div>
          <div class="form-group">
            <label>密码</label>
            <div class="input-wrapper">
              <i class="fa-solid fa-lock"></i>
              <input type="password" v-model="registerForm.password" placeholder="设置密码" />
            </div>
          </div>
          <div class="form-group">
            <label>确认密码</label>
            <div class="input-wrapper">
              <i class="fa-solid fa-check-double"></i>
              <input type="password" v-model="registerForm.confirm" placeholder="再次输入密码" />
            </div>
          </div>
          
          <button class="primary-btn register-btn" @click="handleRegister">注 册</button>
          
          <div class="auth-footer">
            已有账号？ <a href="#" @click.prevent="currentTab = 'login'">返回登录</a>
          </div>
        </template>
        
      </div>
    </div>

    <!-- ================= 已登录状态：个人信息面板 ================= -->
    <div v-else class="profile-container">
      <div class="profile-card glass-panel">
        
        <div class="profile-header">
          <div class="avatar-large">{{ userInfo.avatar }}</div>
          <div class="user-titles">
            <h2>{{ userInfo.nickname }}</h2>
            <span class="role-badge"><i class="fa-solid fa-shield-halved"></i> 认证用户</span>
          </div>
          <button class="logout-btn" @click="handleLogout" title="退出登录">
            <i class="fa-solid fa-right-from-bracket"></i>
          </button>
        </div>

        <div class="profile-content">
          <div class="section-title">
            <h3><i class="fa-solid fa-address-card"></i> 账号信息</h3>
            <button v-if="!isEditing" class="edit-btn" @click="isEditing = true">
              <i class="fa-solid fa-pen-to-square"></i> 编辑资料
            </button>
          </div>

          <div class="info-grid">
            <!-- Username (不可修改) -->
            <div class="info-item">
              <label>登录账号 (不可修改)</label>
              <div class="info-value disabled">{{ userInfo.username }}</div>
            </div>

            <!-- ID (不可修改) -->
            <div class="info-item">
              <label>系统 ID</label>
              <div class="info-value disabled">{{ userInfo.id }}</div>
            </div>

            <!-- Nickname -->
            <div class="info-item">
              <label>显示昵称</label>
              <div v-if="!isEditing" class="info-value">{{ userInfo.nickname }}</div>
              <input v-else type="text" class="edit-input" v-model="editForm.nickname" />
            </div>

            <!-- Email -->
            <div class="info-item">
              <label>绑定邮箱</label>
              <div v-if="!isEditing" class="info-value">{{ userInfo.email || '未绑定邮箱' }}</div>
              <input v-else type="email" class="edit-input" v-model="editForm.email" />
            </div>
          </div>

          <div v-if="isEditing" class="edit-actions">
            <button class="secondary-btn" @click="cancelEdit">取消</button>
            <button class="primary-btn save-btn" @click="saveProfile">保存修改</button>
          </div>
        </div>

        <!-- 可以放一些用户的统计信息 -->
        <div class="stats-section">
          <div class="stat-box">
            <div class="stat-num">12</div>
            <div class="stat-label">历史检测数</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">3</div>
            <div class="stat-label">发起的纠纷</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">98%</div>
            <div class="stat-label">AI 鉴定准确率</div>
          </div>
        </div>

      </div>
    </div>
    
  </div>
</template>

<style scoped>
.profile-layout {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 100px);
  margin-top: 80px;
  padding: 20px;
}

/* 公共毛玻璃面板样式 */
.glass-panel {
  background: rgba(15, 23, 42, 0.6);
  backdrop-filter: blur(16px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
  padding: 40px;
  width: 100%;
}

/* ================= 认证(登录/注册)卡片 ================= */
.auth-container {
  width: 100%;
  max-width: 420px;
}
.auth-header {
  text-align: center;
  margin-bottom: 30px;
}
.auth-header i {
  font-size: 2.5rem;
  color: #60a5fa;
  margin-bottom: 15px;
}
.auth-header h2 {
  color: #f1f5f9;
  margin: 0 0 8px 0;
  font-size: 1.8rem;
}
.auth-header p {
  color: #94a3b8;
  margin: 0;
  font-size: 0.9rem;
}

.form-group {
  margin-bottom: 20px;
}
.form-group label {
  display: block;
  color: #cbd5e1;
  margin-bottom: 8px;
  font-size: 0.9rem;
}
.input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}
.input-wrapper i {
  position: absolute;
  left: 15px;
  color: #64748b;
}
.input-wrapper input {
  width: 100%;
  background: rgba(0, 0, 0, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 12px 12px 12px 40px;
  color: #f8fafc;
  font-size: 1rem;
  transition: border-color 0.2s;
  box-sizing: border-box;
}
.input-wrapper input:focus {
  outline: none;
  border-color: #60a5fa;
  background: rgba(0, 0, 0, 0.3);
}

.primary-btn {
  width: 100%;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  color: white;
  border: none;
  padding: 12px;
  border-radius: 8px;
  font-size: 1.05rem;
  font-weight: bold;
  cursor: pointer;
  transition: transform 0.2s, box-shadow 0.2s;
  margin-top: 10px;
}
.primary-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
}
.register-btn {
  background: linear-gradient(135deg, #8b5cf6, #6366f1);
}

.auth-footer {
  text-align: center;
  margin-top: 25px;
  color: #94a3b8;
  font-size: 0.9rem;
}
.auth-footer a {
  color: #60a5fa;
  text-decoration: none;
  font-weight: bold;
}
.auth-footer a:hover {
  text-decoration: underline;
}

/* ================= 个人信息面板 ================= */
.profile-container {
  width: 100%;
  max-width: 800px;
}

.profile-header {
  display: flex;
  align-items: center;
  gap: 25px;
  padding-bottom: 30px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  position: relative;
}

.avatar-large {
  width: 80px;
  height: 80px;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 2.5rem;
  font-weight: bold;
  color: white;
  box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.user-titles h2 {
  margin: 0 0 10px 0;
  color: #f1f5f9;
  font-size: 1.8rem;
}

.role-badge {
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: 0.85rem;
  border: 1px solid rgba(16, 185, 129, 0.3);
}

.logout-btn {
  position: absolute;
  right: 0;
  top: 0;
  background: rgba(239, 68, 68, 0.1);
  color: #ef4444;
  border: 1px solid rgba(239, 68, 68, 0.3);
  width: 40px;
  height: 40px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 1.1rem;
}
.logout-btn:hover {
  background: #ef4444;
  color: white;
}

.profile-content {
  margin-top: 30px;
}

.section-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}
.section-title h3 {
  color: #e2e8f0;
  margin: 0;
  font-size: 1.2rem;
}
.edit-btn {
  background: rgba(255,255,255,0.05);
  color: #cbd5e1;
  border: 1px solid rgba(255,255,255,0.1);
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.2s;
}
.edit-btn:hover {
  background: rgba(255,255,255,0.1);
}

.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.info-item {
  background: rgba(0, 0, 0, 0.15);
  padding: 15px 20px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.03);
}

.info-item label {
  display: block;
  color: #64748b;
  font-size: 0.85rem;
  margin-bottom: 8px;
}

.info-value {
  color: #e2e8f0;
  font-size: 1.05rem;
  font-weight: 500;
}
.info-value.disabled {
  color: #94a3b8;
}

.edit-input {
  width: 100%;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid #60a5fa;
  color: white;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 1rem;
  box-sizing: border-box;
}
.edit-input:focus {
  outline: none;
  box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.3);
}

.edit-actions {
  margin-top: 25px;
  display: flex;
  justify-content: flex-end;
  gap: 15px;
}
.secondary-btn {
  background: transparent;
  color: #cbd5e1;
  border: 1px solid rgba(255,255,255,0.2);
  padding: 10px 20px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}
.secondary-btn:hover {
  background: rgba(255,255,255,0.05);
}
.save-btn {
  width: auto;
  margin-top: 0;
  padding: 10px 25px;
}

/* 统计数据 */
.stats-section {
  margin-top: 40px;
  display: flex;
  gap: 20px;
}
.stat-box {
  flex: 1;
  background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(0,0,0,0.2) 100%);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 12px;
  padding: 20px;
  text-align: center;
}
.stat-num {
  font-size: 2rem;
  font-weight: bold;
  color: #60a5fa;
  margin-bottom: 5px;
}
.stat-label {
  color: #94a3b8;
  font-size: 0.85rem;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .info-grid {
    grid-template-columns: 1fr;
  }
  .stats-section {
    flex-wrap: wrap;
  }
  .stat-box {
    flex: 1 1 calc(50% - 10px);
  }
}
</style>
