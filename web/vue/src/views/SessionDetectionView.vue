<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick, computed } from 'vue';
import { useRouter } from 'vue-router';
import { fetchSessions, fetchSessionMessages, sendTextMessage, sendImageMessage, detectMessage, explainLogicStream, createSession, updateSession, deleteSession, joinSession } from '../api';
import { useUserStore } from '../stores/user';
import FloatingAssistant from '../components/session/FloatingAssistant.vue';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({ breaks: true, gfm: true });
const renderMd = (text: string): string => {
  try { return DOMPurify.sanitize(marked.parse(text) as string); }
  catch { return text; }
};

const { state: userState } = useUserStore();
const router = useRouter();

const sessions = ref<any[]>([]);
const activeSession = ref<any>(null);
const messages = ref<any[]>([]);
const showAiPanel = ref(false);
const isJudging = ref(false);
const inputText = ref('');
const fileInput = ref<HTMLInputElement | null>(null);
const messageListEl = ref<HTMLElement | null>(null);

// 会话管理 UI
const showAddSession = ref(false);
const newSessionTitle = ref('');
const addSessionInputEl = ref<HTMLInputElement | null>(null);
const showSessionMenu = ref(false);
const isRenamingSession = ref(false);
const renameTitle = ref('');
// 邀请 & 加入
const inviteSessionId = ref('');       // 刚创建后弹出邀请码
const showInviteModal = ref(false);
const showJoinInput = ref(false);      // 加入会话输入条
const joinCode = ref('');
const joinInputEl = ref<HTMLInputElement | null>(null);
const joinError = ref('');

// AI 检测结果数据
const detectResult = ref<any>(null);
const detectingMsgId = ref<string | null>(null);
const useVlOption = ref(false); // 用户可选是否启用 VL 分析
const vlReport = ref('');
const vlStreaming = ref(false);

// 检测步骤状态
const stepLaft = ref<'waiting' | 'active' | 'done'>('waiting');
const stepVl = ref<'waiting' | 'active' | 'done' | 'skipped'>('waiting');
const stepReport = ref<'waiting' | 'active' | 'done'>('waiting');

const isFake = computed(() => detectResult.value && detectResult.value.class_name !== 'Real');

const scrollToBottom = async () => {
  await nextTick();
  if (messageListEl.value) {
    messageListEl.value.scrollTop = messageListEl.value.scrollHeight;
  }
};

const loadSessions = async () => {
  try {
    const res = await fetchSessions();
    if (res.success && res.sessions.length > 0) {
      sessions.value = res.sessions;
      activeSession.value = sessions.value[0];
      await loadMessages(activeSession.value.id);
    }
  } catch (error) {
    console.error("加载会话失败:", error);
  }
};

const lastMsgId = ref<string | null>(null);
const pollingTimer = ref<any>(null);
const sessionPollTimer = ref<any>(null);

const loadMessages = async (sessionId: string) => {
  try {
    const res = await fetchSessionMessages(sessionId);
    if (res.success) {
      messages.value = res.messages;
      lastMsgId.value = res.messages.length > 0 ? res.messages[res.messages.length - 1].id : null;
      await nextTick();
      scrollToBottom();
    }
  } catch (error) {
    console.error("加载消息失败:", error);
  }
};

const pollMessages = async () => {
  if (!activeSession.value) return;
  try {
    const res = await fetchSessionMessages(activeSession.value.id, lastMsgId.value ?? undefined);
    if (res.success && res.messages.length > 0) {
      const newIds = new Set(messages.value.map((m: any) => m.id));
      const fresh = res.messages.filter((m: any) => !newIds.has(m.id));
      if (fresh.length > 0) {
        messages.value.push(...fresh);
        lastMsgId.value = fresh[fresh.length - 1].id;
        await nextTick();
        scrollToBottom();
      }
    }
  } catch { /* 轮询失败静默忽略 */ }
};

const pollSessions = async () => {
  try {
    const res = await fetchSessions();
    if (res.success) {
      const existingIds = new Set(sessions.value.map((s: any) => s.id));
      const fresh = res.sessions.filter((s: any) => !existingIds.has(s.id));
      // 更新已有会话的信息（desc/status）
      sessions.value = sessions.value.map((s: any) => {
        const updated = res.sessions.find((r: any) => r.id === s.id);
        return updated ? { ...s, desc: updated.desc, status: updated.status } : s;
      });
      if (fresh.length > 0) sessions.value.push(...fresh);
    }
  } catch { /* 轮询失败静默忽略 */ }
};

const startPolling = () => {
  stopPolling();
  pollingTimer.value = setInterval(pollMessages, 1500);
  sessionPollTimer.value = setInterval(pollSessions, 5000);
};

const stopPolling = () => {
  if (pollingTimer.value) { clearInterval(pollingTimer.value); pollingTimer.value = null; }
  if (sessionPollTimer.value) { clearInterval(sessionPollTimer.value); sessionPollTimer.value = null; }
};

onMounted(() => {
  loadSessions().then(() => startPolling());
  document.addEventListener('click', () => { showSessionMenu.value = false; });
  document.addEventListener('visibilitychange', () => {
    document.hidden ? stopPolling() : startPolling();
  });
});

onUnmounted(() => {
  stopPolling();
});

const handleSessionClick = async (item: any) => {
  activeSession.value = item;
  isRenamingSession.value = false;
  lastMsgId.value = null;
  stopPolling();
  await loadMessages(item.id);
  startPolling();
};

// 新建会话
const handleAddBtnClick = async () => {
  showAddSession.value = !showAddSession.value;
  if (showAddSession.value) {
    await nextTick();
    addSessionInputEl.value?.focus();
  }
};

const confirmAddSession = async () => {
  const title = newSessionTitle.value.trim();
  if (!title) return;
  try {
    const res = await createSession(title);
    if (res.success) {
      sessions.value.unshift(res.session);
      activeSession.value = res.session;
      messages.value = [];
      // 创建成功后展示邀请码
      inviteSessionId.value = res.session.id;
      showInviteModal.value = true;
    }
  } catch (e) { console.error('创建会话失败:', e); }
  newSessionTitle.value = '';
  showAddSession.value = false;
};

// 加入会话
const handleJoinBtnClick = async () => {
  showJoinInput.value = !showJoinInput.value;
  showAddSession.value = false;
  joinError.value = '';
  if (showJoinInput.value) {
    await nextTick();
    joinInputEl.value?.focus();
  }
};

const confirmJoinSession = async () => {
  const code = joinCode.value.trim();
  if (!code) return;
  joinError.value = '';
  try {
    const res = await joinSession(code);
    if (res.success) {
      // 如果列表中已有就不重复添加
      const exists = sessions.value.find((s: any) => s.id === res.session.id);
      if (!exists) sessions.value.unshift(res.session);
      activeSession.value = res.session.id ? sessions.value.find((s: any) => s.id === res.session.id) : res.session;
      await loadMessages(res.session.id);
      showJoinInput.value = false;
      joinCode.value = '';
    } else {
      joinError.value = res.message || '加入失败';
    }
  } catch (e: any) {
    joinError.value = e.response?.data?.message || '无效的邀请码';
  }
};

// 复制邀请码
const copyInviteCode = () => {
  navigator.clipboard.writeText(inviteSessionId.value);
};

// 重命名会话
const startRenameSession = () => {
  if (!activeSession.value) return;
  renameTitle.value = activeSession.value.title;
  isRenamingSession.value = true;
  showSessionMenu.value = false;
  nextTick(() => {
    const el = document.querySelector<HTMLInputElement>('.rename-input');
    el?.focus();
    el?.select();
  });
};

const confirmRename = async () => {
  const title = renameTitle.value.trim();
  if (!title || !activeSession.value) { isRenamingSession.value = false; return; }
  try {
    const res = await updateSession(activeSession.value.id, title);
    if (res.success) {
      const idx = sessions.value.findIndex((s: any) => s.id === activeSession.value.id);
      if (idx !== -1) sessions.value[idx].title = title;
      activeSession.value = { ...activeSession.value, title };
    }
  } catch (e) { console.error('重命名失败:', e); }
  isRenamingSession.value = false;
};

// 删除会话
const handleDeleteSession = async () => {
  if (!activeSession.value) return;
  if (!confirm(`确认删除「${activeSession.value.title}」？此操作不可恢复。`)) return;
  showSessionMenu.value = false;
  try {
    const res = await deleteSession(activeSession.value.id);
    if (res.success) {
      sessions.value = sessions.value.filter((s: any) => s.id !== activeSession.value.id);
      activeSession.value = sessions.value[0] || null;
      messages.value = [];
      if (activeSession.value) await loadMessages(activeSession.value.id);
    }
  } catch (e) { console.error('删除会话失败:', e); }
};

const requestAiDetection = async (msgId: string) => {
  showAiPanel.value = true;
  isJudging.value = true;
  detectResult.value = null;
  detectingMsgId.value = msgId;
  vlReport.value = '';
  vlStreaming.value = false;

  // 步骤状态初始化
  stepLaft.value = 'active';
  stepVl.value = useVlOption.value ? 'waiting' : 'skipped';
  stepReport.value = 'waiting';

  try {
    // 调用 LaFT 检测（后端同时可选 VL）
    const res = await detectMessage(msgId, useVlOption.value);
    stepLaft.value = 'done';

    if (res.success) {
      detectResult.value = res.detect;

      // 标记消息已鉴定
      const msg = messages.value.find((m: any) => m.id === msgId);
      if (msg) msg.hasBeenDetected = true;

      if (useVlOption.value && res.detect.vl_report) {
        stepVl.value = 'done';
        vlReport.value = res.detect.vl_report;
      } else if (useVlOption.value) {
        stepVl.value = 'done';
      }

      stepReport.value = 'done';
      isJudging.value = false;
    } else {
      isJudging.value = false;
      detectResult.value = { error: res.message || '检测失败' };
    }
  } catch (e: any) {
    console.error("AI检测失败:", e);
    stepLaft.value = 'done';
    isJudging.value = false;
    detectResult.value = { error: e.response?.data?.message || '检测请求失败' };
  }
};

// 单独调用 VL 流式分析（在已有检测结果后）
const startVlStream = async () => {
  if (!detectingMsgId.value || !detectResult.value) return;
  const msg = messages.value.find((m: any) => m.id === detectingMsgId.value);
  if (!msg || msg.type !== 'image') return;

  vlReport.value = '';
  vlStreaming.value = true;

  try {
    // 需要把图片转 base64 发给 VL
    const imgUrl = msg.content;
    const response = await fetch(imgUrl);
    const blob = await response.blob();
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64 = (reader.result as string).split(',')[1];
      await explainLogicStream(base64, isFake.value, (chunk: string) => {
        vlReport.value += chunk;
      });
      vlStreaming.value = false;
    };
    reader.readAsDataURL(blob);
  } catch (e) {
    console.error("VL流式分析失败:", e);
    vlReport.value += '\n[分析中断]';
    vlStreaming.value = false;
  }
};

const handleSendText = async () => {
  if (!inputText.value.trim() || !activeSession.value) return;
  try {
    const res = await sendTextMessage(activeSession.value.id, inputText.value.trim());
    if (res.success) {
      messages.value.push(res.message);
      inputText.value = '';
      scrollToBottom();
    }
  } catch (e) {
    console.error("发送消息失败:", e);
  }
};

const triggerImageUpload = () => {
  fileInput.value?.click();
};

const handleImageUpload = async (event: Event) => {
  const target = event.target as HTMLInputElement;
  const file = target.files?.[0];
  if (!file || !activeSession.value) return;
  try {
    const res = await sendImageMessage(activeSession.value.id, file);
    if (res.success) {
      messages.value.push(res.message);
      scrollToBottom();
    }
  } catch (e) {
    console.error("发送图片失败:", e);
  }
  target.value = '';
};

</script>

<template>
  <div class="page-wrapper">
    <div class="chat-container">
      
      <!-- 左侧：纠纷会话列表 -->
      <div class="sidebar">
        <div class="sidebar-header">
          <h3><i class="fa-solid fa-scale-balanced" style="margin-right: 8px;"></i>纠纷法庭中心</h3>
          <div class="header-btns">
            <button class="header-icon-btn" @click.stop="handleJoinBtnClick" title="加入纠纷">
              <i class="fa-solid fa-link"></i>
            </button>
            <button class="add-btn" @click.stop="handleAddBtnClick"><i class="fa-solid fa-plus"></i></button>
          </div>
        </div>
        <div v-if="showJoinInput" class="add-session-row join-row">
          <i class="fa-solid fa-hashtag join-hash"></i>
          <input ref="joinInputEl" type="text" v-model="joinCode" placeholder="粘贴邀请码加入会话..." @keyup.enter="confirmJoinSession" @keyup.esc="showJoinInput = false" />
          <button class="add-session-confirm" @click="confirmJoinSession" title="加入"><i class="fa-solid fa-arrow-right"></i></button>
          <button class="add-session-cancel" @click="showJoinInput = false" title="取消"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <p v-if="joinError" class="join-error"><i class="fa-solid fa-triangle-exclamation"></i> {{ joinError }}</p>
        <div v-if="showAddSession" class="add-session-row">
          <input ref="addSessionInputEl" type="text" v-model="newSessionTitle" placeholder="输入纠纷标题..." @keyup.enter="confirmAddSession" @keyup.esc="showAddSession = false" />
          <button class="add-session-confirm" @click="confirmAddSession" title="确认"><i class="fa-solid fa-check"></i></button>
          <button class="add-session-cancel" @click="showAddSession = false" title="取消"><i class="fa-solid fa-xmark"></i></button>
        </div>
        <div class="search-bar">
          <i class="fa-solid fa-search"></i>
          <input type="text" placeholder="搜索纠纷记录..." />
        </div>
        <!-- 未登录提示 -->
        <div v-if="!userState.isLoggedIn" class="not-logged-in-tip">
          <i class="fa-solid fa-lock"></i>
          <p>请先登录后查看纠纷会话</p>
          <button class="login-tip-btn" @click="router.push('/profile')"><i class="fa-solid fa-right-to-bracket"></i> 去登录</button>
        </div>

        <ul v-else class="session-list">
          <li 
            v-for="item in sessions" 
            :key="item.id" 
            :class="['session-item', { active: activeSession && activeSession.id === item.id }]"
            @click="handleSessionClick(item)"
          >
            <div class="session-info">
              <span class="session-title">{{ item.title }}</span>
              <span class="session-desc">{{ item.desc || '暂无详细描述...' }}</span>
            </div>
            <div class="session-meta">
              <span class="session-time">{{ item.time }}</span>
              <span v-if="item.status === 'ai_judging'" class="badge judging"><i class="fa-solid fa-robot"></i> 介入中</span>
              <span v-if="item.status === 'open'" class="badge open"><i class="fa-regular fa-clock"></i> 未解决</span>
            </div>
          </li>
        </ul>
      </div>

      <!-- 中间：聊天/辩论流 -->
      <div class="chat-main chat-empty" v-if="!userState.isLoggedIn">
        <div class="chat-empty-state">
          <i class="fa-solid fa-scale-balanced"></i>
          <h3>纠纷法庭中心</h3>
          <p>登录后可发起或参与纠纷仲裁会话，上传证据图片并使用 AI 进行司法鉴定。</p>
          <button class="primary-btn" @click="router.push('/profile')"><i class="fa-solid fa-right-to-bracket"></i> 立即登录</button>
        </div>
      </div>
      <div class="chat-main chat-empty" v-else-if="!activeSession && sessions.length === 0">
        <div class="chat-empty-state">
          <i class="fa-solid fa-folder-open"></i>
          <h3>暂无纠纷会话</h3>
          <p>点击左侧 <strong>+</strong> 新建一个纠纷会话，或使用 <i class="fa-solid fa-link"></i> 通过邀请码加入。</p>
        </div>
      </div>
      <div class="chat-main" v-else-if="activeSession">
        <div class="chat-header">
          <div class="header-left">
            <template v-if="isRenamingSession">
              <div class="rename-wrapper">
                <input class="rename-input" v-model="renameTitle" @keyup.enter="confirmRename" @keyup.esc="isRenamingSession = false" placeholder="输入新标题..." />
                <button class="rename-confirm-btn" @click="confirmRename" title="确认"><i class="fa-solid fa-check"></i></button>
                <button class="rename-confirm-btn" @click="isRenamingSession = false" title="取消" style="color:#94a3b8"><i class="fa-solid fa-xmark"></i></button>
              </div>
            </template>
            <h3 v-else>{{ activeSession.title }}</h3>
            <span class="status-text"><span class="pulse-dot"></span> 申诉中</span>
          </div>
          <div class="header-actions">
            <div class="menu-wrapper" @click.stop>
              <button class="icon-btn" @click="showSessionMenu = !showSessionMenu"><i class="fa-solid fa-ellipsis-vertical"></i></button>
              <div v-if="showSessionMenu" class="session-menu">
                <button class="menu-item" @click="startRenameSession"><i class="fa-solid fa-pen"></i> 编辑标题</button>
                <button class="menu-item danger" @click="handleDeleteSession"><i class="fa-solid fa-trash"></i> 删除会话</button>
              </div>
            </div>
          </div>
        </div>
        
        <div class="message-list" ref="messageListEl">
          <div class="date-divider"><span>今天 09:58</span></div>
          
          <div 
            v-for="msg in messages" 
            :key="msg.id" 
            :class="['message-wrapper', msg.role, { self: msg.sender_id === userState.id }]"
          >
            <div v-if="msg.role === 'system'" class="system-message">
              <span><i class="fa-solid fa-info-circle"></i> {{ msg.content }}</span>
            </div>
            
            <template v-else>
              <div class="avatar-container">
                <div class="avatar">
                  <img v-if="msg.avatar && msg.avatar.startsWith('/')" :src="msg.avatar" class="avatar-img" alt="" />
                  <span v-else>{{ msg.avatar || msg.name[0] }}</span>
                </div>
              </div>
              <div class="message-content-wrapper">
                <div class="sender-name">{{ msg.name }}</div>
                
                <!-- 文本消息 -->
                <div class="message-bubble" v-if="msg.type === 'text'">{{ msg.content }}</div>

                <!-- 图片消息 -->
                <div class="image-bubble" v-else-if="msg.type === 'image'">
                  <img :src="msg.content" alt="纠纷会话图片" />
                  <div class="image-actions">
                    <button class="detect-btn" @click="requestAiDetection(msg.id)" :disabled="detectingMsgId === msg.id && isJudging">
                      <span class="btn-content">
                        <i class="fa-solid fa-microscope"></i>
                        {{ (detectingMsgId === msg.id && isJudging) ? '鉴定中…' : (msg.hasBeenDetected ? '重新鉴定' : 'AI 入场鉴定') }}
                      </span>
                      <span v-if="msg.hasBeenDetected && !(detectingMsgId === msg.id && isJudging)" class="detected-badge" title="已完成过一次鉴定">
                        <i class="fa-solid fa-check-circle"></i>
                      </span>
                    </button>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </div>

        <div class="chat-input-area">
          <input type="file" ref="fileInput" accept="image/*" style="display:none" @change="handleImageUpload" />
          <button class="upload-btn" title="上传证据图片" @click="triggerImageUpload"><i class="fa-solid fa-image"></i></button>
          <div class="input-wrapper">
            <input type="text" v-model="inputText" placeholder="输入请求" @keyup.enter="handleSendText" />
            <button class="emoji-btn"><i class="fa-regular fa-face-smile"></i></button>
          </div>
          <button class="send-btn" title="发送" @click="handleSendText"><i class="fa-solid fa-paper-plane"></i></button>
        </div>
      </div>

      <!-- 右侧：AI 裁决面板 (条件渲染) -->
      <transition name="slide-panel">
        <div class="ai-panel" v-if="showAiPanel">
          <div class="ai-header">
            <div class="title-with-icon">
              <div class="icon-box"><i class="fa-solid fa-gavel"></i></div>
              <span>AI 鉴定法庭</span>
            </div>
            <button class="close-btn" @click="showAiPanel = false"><i class="fa-solid fa-xmark"></i></button>
          </div>
          
          <div class="ai-content">
            <div v-if="isJudging" class="judging-state">
              <div class="scanning-animation">
                <div class="scan-line"></div>
                <i class="fa-solid fa-microchip ai-core-icon"></i>
              </div>
              <h3 class="judging-title">正在多维取证...</h3>
              <ul class="judging-steps">
                <li :class="['step', stepLaft]">
                  <i :class="stepLaft === 'done' ? 'fa-solid fa-check' : stepLaft === 'active' ? 'fa-solid fa-spinner fa-spin' : 'fa-regular fa-circle'"></i>
                  提取图像频域特征 (LaRE 模型)
                </li>
                <li :class="['step', stepVl]">
                  <i :class="stepVl === 'done' ? 'fa-solid fa-check' : stepVl === 'active' ? 'fa-solid fa-spinner fa-spin' : stepVl === 'skipped' ? 'fa-solid fa-forward' : 'fa-regular fa-circle'"></i>
                  审阅上下文意图 (Qwen-VL)
                  <span v-if="stepVl === 'skipped'" class="step-note">已跳过</span>
                </li>
                <li :class="['step', stepReport]">
                  <i :class="stepReport === 'done' ? 'fa-solid fa-check' : stepReport === 'active' ? 'fa-solid fa-spinner fa-spin' : 'fa-regular fa-circle'"></i>
                  生成法庭判定报告
                </li>
              </ul>
            </div>
            <!-- VL 开关 -->
            <div class="vl-toggle" v-if="isJudging || !detectResult">
              <label class="toggle-label">
                <input type="checkbox" v-model="useVlOption" />
                <span class="toggle-text"><i class="fa-solid fa-brain"></i> 启用 Qwen-VL 多模态报告</span>
              </label>
            </div>
            <div v-else-if="detectResult && !detectResult.error" class="result-state fade-in">
                <div :class="['status-card', isFake ? 'fake' : 'real']">
                  <div class="status-icon">
                    <i :class="isFake ? 'fa-solid fa-triangle-exclamation' : 'fa-solid fa-shield-check'"></i>
                  </div>
                  <div class="status-info">
                    <h4>{{ isFake ? 'AI生成 (证据确凿)' : '真实图片 (未检出伪造)' }}</h4>
                    <span class="confidence">置信度: {{ (detectResult.confidence * 100).toFixed(1) }}%</span>
                  </div>
                </div>

                <!-- 概率分布 -->
                <div class="panel-section" v-if="detectResult.probabilities">
                  <div class="section-title">
                    <i class="fa-solid fa-chart-bar"></i> 概率分布
                  </div>
                  <div class="prob-bars">
                    <div class="prob-item" v-for="(prob, label) in detectResult.probabilities" :key="label">
                      <div class="prob-label">{{ label }}</div>
                      <div class="prob-bar-bg">
                        <div class="prob-bar-fill" :style="{ width: (prob * 100) + '%' }" :class="{ highlight: prob > 0.5 }"></div>
                      </div>
                      <div class="prob-value">{{ (prob * 100).toFixed(1) }}%</div>
                    </div>
                  </div>
                </div>
                
                <!-- 热力图 -->
                <div class="panel-section" v-if="detectResult.heatmap">
                  <div class="section-title">
                    <i class="fa-solid fa-map"></i> 伪影热力图分析
                  </div>
                  <div class="heatmap-placeholder">
                    <img :src="'data:image/jpeg;base64,' + detectResult.heatmap" alt="Heatmap" />
                  </div>
                </div>

                <!-- VL 报告区 -->
                <div class="panel-section">
                  <div class="section-title">
                    <i class="fa-solid fa-scroll"></i> 多模态分析报告
                    <button v-if="!vlReport && !vlStreaming && !detectResult.vl_report" class="vl-trigger-btn" @click="startVlStream">
                      <i class="fa-solid fa-brain"></i> 生成 VL 报告
                    </button>
                  </div>
                  <div v-if="vlReport || vlStreaming" class="logic-report">
                    <div class="report-block vl-block">
                      <span class="report-tag visual">Qwen-VL</span>
                      <div class="vl-text markdown-body" v-html="renderMd(vlReport)"></div>
                      <span v-if="vlStreaming" class="cursor-blink">|</span>
                    </div>
                  </div>
                  <div v-else-if="detectResult.vl_report" class="logic-report">
                    <div class="report-block vl-block">
                      <span class="report-tag visual">Qwen-VL</span>
                      <div class="vl-text markdown-body" v-html="renderMd(detectResult.vl_report)"></div>
                    </div>
                  </div>
                  <div v-else class="vl-placeholder">
                    <p>点击上方按钮调用 Qwen-VL 生成详细多模态分析报告</p>
                  </div>
                </div>
                
                <div class="verdict-box">
                  <div class="verdict-title"><i class="fa-solid fa-gavel"></i> 判决建议</div>
                  <p v-if="isFake">该图片被 AI 模型高置信度判定为 <strong>AIGC 生成图像</strong>，建议支持买家诉求。</p>
                  <p v-else>该图片未检出明显伪造痕迹，被判定为 <strong>真实图像</strong>。</p>
                </div>
            </div>

            <!-- 检测失败 -->
            <div v-else-if="detectResult && detectResult.error" class="result-state fade-in">
              <div class="status-card error">
                <div class="status-icon"><i class="fa-solid fa-circle-exclamation"></i></div>
                <div class="status-info">
                  <h4>检测失败</h4>
                  <span class="confidence">{{ detectResult.error }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </transition>
    </div>
  </div>

  <!-- 邀请码弹窗 -->
  <teleport to="body">
    <transition name="fade">
      <div v-if="showInviteModal" class="invite-mask" @click.self="showInviteModal = false">
        <div class="invite-modal">
          <div class="invite-modal-header">
            <i class="fa-solid fa-paper-plane-top"></i>
            <h3>邀请对方加入</h3>
            <button class="invite-close" @click="showInviteModal = false"><i class="fa-solid fa-xmark"></i></button>
          </div>
          <p class="invite-tip">将以下邀请码发送给对方，对方点击左侧&nbsp;<i class="fa-solid fa-link"></i>&nbsp;输入后即可加入此纠纷会话。</p>
          <div class="invite-code-box">
            <span class="invite-code">{{ inviteSessionId }}</span>
            <button class="copy-btn" @click="copyInviteCode" title="复制">
              <i class="fa-solid fa-copy"></i> 复制
            </button>
          </div>
          <button class="invite-ok-btn" @click="showInviteModal = false">我已发送，关闭</button>
        </div>
      </div>
    </transition>
  </teleport>

  <!-- 悬浮 AI 小助手 -->
  <FloatingAssistant
    :session-id="activeSession ? activeSession.id : null"
    :session-title="activeSession ? activeSession.title : ''"
  />
</template>

<style scoped>
.page-wrapper {
  padding-top: 80px;
  min-height: 100vh;
  background: radial-gradient(circle at 50% 0%, var(--card-bg-solid) 0%, var(--panel-bg-solid) 100%);
  display: flex;
  justify-content: center;
}

.chat-container {
  display: flex;
  height: calc(100vh - 120px);
  width: 100vw;
  max-width: 1300px;
  background: var(--panel-bg);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border-color-light);
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px var(--border-color-light) inset;
}

/* 左侧列表 */
.sidebar {
  width: 320px;
  background: var(--panel-bg);
  border-right: 1px solid var(--border-color-light);
  display: flex;
  flex-direction: column;
  z-index: 10;
}
.sidebar-header {
  padding: 24px 20px 16px;
  color: var(--text-main);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.sidebar-header h3 {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
  letter-spacing: 0.5px;
}
.header-btns { display: flex; align-items: center; gap: 8px; }
.header-icon-btn {
  background: var(--border-color-light); border: 1px solid var(--border-color);
  color: var(--text-muted); width: 32px; height: 32px; border-radius: 8px;
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  transition: all 0.2s; font-size: 0.9rem;
}
.header-icon-btn:hover { color: var(--text-main); background: var(--border-color); }
.add-btn {
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border: none;
  color: white;
  width: 32px; height: 32px; 
  border-radius: 8px; 
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}
.add-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
}

.search-bar {
  margin: 0 20px 16px;
  position: relative;
}
.search-bar i {
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-muted);
  font-size: 0.9rem;
}
.search-bar input {
  width: 100%;
  background: var(--border-color-light);
  border: 1px solid var(--border-color-light);
  border-radius: 10px;
  padding: 10px 12px 10px 36px;
  color: var(--text-main);
  font-size: 0.9rem;
  transition: all 0.2s;
}
.search-bar input:focus {
  background: var(--border-color-light);
  border-color: rgba(59, 130, 246, 0.5);
  outline: none;
}

.session-list {
  list-style: none; padding: 0; margin: 0; overflow-y: auto;
  flex: 1;
}
/* 自定义滚动条 */
.session-list::-webkit-scrollbar, .message-list::-webkit-scrollbar, .ai-content::-webkit-scrollbar {
  width: 6px;
}
.session-list::-webkit-scrollbar-thumb, .message-list::-webkit-scrollbar-thumb, .ai-content::-webkit-scrollbar-thumb {
  background: var(--border-color);
  border-radius: 3px;
}

.session-item {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color-light);
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}
.session-item:hover { background: var(--border-color-light); }
.session-item.active { 
  background: linear-gradient(90deg, rgba(59, 130, 246, 0.15) 0%, rgba(59, 130, 246, 0.05) 100%);
  border-left: 3px solid #3b82f6; 
}
.session-info { flex: 1; min-width: 0; }
.session-title { 
  color: var(--text-main); 
  font-size: 0.95rem; 
  font-weight: 500; 
  display: block; 
  overflow: hidden; 
  text-overflow: ellipsis; 
  white-space: nowrap; 
  margin-bottom: 6px;
}
.session-desc {
  color: var(--text-muted);
  font-size: 0.8rem;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.session-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
  flex-shrink: 0;
}
.session-time { color: var(--text-muted); font-size: 0.75rem; font-weight: 500;}
.badge { font-size: 0.7rem; padding: 3px 8px; border-radius: 6px; font-weight: 500; display: inline-flex; align-items: center; gap: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
.badge.judging { background: linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(139, 92, 246, 0.2)); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
.badge.open { background: var(--border-color); color: var(--text-muted); border: 1px solid var(--border-color);}

/* 未登录提示 */
.not-logged-in-tip {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 24px 16px;
  text-align: center;
  color: var(--text-muted);
}
.not-logged-in-tip i { font-size: 2rem; color: #475569; }
.not-logged-in-tip p { font-size: 0.85rem; line-height: 1.5; }
.login-tip-btn {
  margin-top: 4px;
  padding: 8px 18px;
  font-size: 0.85rem;
  border: none;
  border-radius: 8px;
  background: linear-gradient(135deg, #8b5cf6, #6366f1);
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s;
  box-shadow: 0 3px 10px rgba(99,102,241,0.3);
}
.login-tip-btn:hover { transform: translateY(-1px); filter: brightness(1.1); }

/* 空状态（无会话） */
.chat-empty {
  align-items: center;
  justify-content: center;
}
.chat-empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  text-align: center;
  color: var(--text-muted);
  max-width: 360px;
  padding: 40px 20px;
}
.chat-empty-state > i { font-size: 3.5rem; color: #334155; }
.chat-empty-state h3 { font-size: 1.15rem; color: var(--text-main); margin: 0; }
.chat-empty-state p { font-size: 0.9rem; line-height: 1.7; }

/* 中间聊天区 */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMiIgY3k9IjIiIHI9IjEiIGZpbGw9InJnYmEoMjU1LDI1NSwyNTUsMC4wMykiLz48L3N2Zz4=') repeat;
}
.chat-header {
  padding: 20px 24px;
  background: var(--panel-bg);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border-color-light);
  display: flex;
  justify-content: space-between;
  align-items: center;
  z-index: 10;
}
.header-left h3 {
  color: var(--text-main);
  margin: 0 0 4px 0;
  font-size: 1.1rem;
}
.pulse-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; margin-right: 6px;
  box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}
.status-text { font-size: 0.8rem; color: var(--text-muted); display: flex; align-items: center;}
.icon-btn { background: transparent; border: none; color: var(--text-muted); font-size: 1.2rem; cursor: pointer; padding: 8px; border-radius: 8px; transition: 0.2s;}
.icon-btn:hover { background: var(--border-color-light); color: white;}

.message-list {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.date-divider {
  text-align: center;
  margin: 10px 0;
}
.date-divider span {
  background: var(--border-color);
  color: var(--text-muted);
  font-size: 0.75rem;
  padding: 4px 12px;
  border-radius: 12px;
}

.message-wrapper { display: flex; gap: 16px; max-width: 70%; }
.message-wrapper.self { align-self: flex-end; flex-direction: row-reverse; }
.message-wrapper:not(.self) { align-self: flex-start; }

.system-message { align-self: center; margin: 15px 0; max-width: 100%;}
.system-message span { 
  background: rgba(59, 130, 246, 0.1); 
  border: 1px solid rgba(59, 130, 246, 0.2);
  color: #3b82f6; 
  padding: 8px 16px; 
  border-radius: 16px; 
  font-size: 0.85rem;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.avatar { 
  width: 40px; height: 40px; 
  border-radius: 12px; 
  background: linear-gradient(135deg, #475569, #334155); 
  color: white; 
  display: flex; align-items: center; justify-content: center; 
  font-weight: 600; font-size: 1.1rem;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2);
  overflow: hidden;
}
.avatar .avatar-img { width: 100%; height: 100%; object-fit: cover; border-radius: 12px; }
.buyer .avatar { background: linear-gradient(135deg, #60a5fa, #3b82f6); }
.seller .avatar { background: linear-gradient(135deg, #fbbf24, #f59e0b); }

.message-content-wrapper { display: flex; flex-direction: column; max-width: calc(100% - 56px); }
.self .message-content-wrapper { align-items: flex-end; }
.sender-name { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px; padding: 0 4px;}

.message-bubble {
  background: var(--card-bg);
  border: 1px solid var(--border-color-light);
  color: var(--text-main);
  padding: 12px 18px;
  border-radius: 4px 16px 16px 16px;
  line-height: 1.6;
  font-size: 0.95rem;
  word-break: break-word;
  box-shadow: 0 4px 15px rgba(0,0,0,0.1);
  width: fit-content;
  max-width: 100%;
}
.self .message-bubble { 
  background: linear-gradient(135deg, #2563eb, #1d4ed8); 
  border: 1px solid var(--border-color);
  color: white; 
  border-radius: 16px 4px 16px 16px; 
  box-shadow: 0 4px 15px rgba(37, 99, 235, 0.2);
}

.image-bubble {
  background: var(--card-bg);
  padding: 12px;
  border-radius: 16px;
  border: 1px solid var(--border-color-light);
  box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}
.image-bubble img { max-width: 280px; border-radius: 8px; display: block; }
.image-actions { margin-top: 12px; display: flex; justify-content: stretch;}
.detect-btn {
  flex: 1;
  background: linear-gradient(135deg, #8b5cf6, var(--primary-color));
  color: white; border: none; padding: 10px 0; border-radius: 8px; font-size: 0.9rem; font-weight: 500; cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); 
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
  display: flex; align-items: center; justify-content: center;
}
.detect-btn:hover:not(:disabled) { 
  transform: translateY(-2px); 
  box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
  filter: brightness(1.1);
}
.detect-btn:active:not(:disabled) { transform: translateY(0); }
.detect-btn:disabled {
  background: #334155;
  color: var(--text-muted);
  cursor: not-allowed;
  box-shadow: none;
}
.detected-badge {
  margin-left: 6px;
  color: #34d399;
  font-size: 0.85rem;
  opacity: 0.85;
}

.chat-input-area {
  padding: 20px 24px;
  background: var(--panel-bg);
  backdrop-filter: blur(10px);
  border-top: 1px solid var(--border-color-light);
  display: flex; gap: 12px;
  align-items: center;
}
.input-wrapper {
  flex: 1;
  position: relative;
  display: flex;
  align-items: center;
}
.chat-input-area input {
  width: 100%;
  height: 48px;
  background: var(--border-color-light); 
  border: 1px solid var(--border-color);
  color: white; 
  border-radius: 24px; 
  padding: 0 45px 0 20px; 
  font-size: 0.95rem;
  outline: none;
  transition: all 0.2s;
}
.chat-input-area input:focus {
  background: var(--border-color-light);
  border-color: rgba(59, 130, 246, 0.5);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
}
.chat-input-area input::placeholder { color: #64748b; }
.emoji-btn {
  position: absolute; right: 12px;
  background: none; border: none; color: var(--text-muted);
  cursor: pointer; font-size: 1.2rem;
  transition: color 0.2s;
}
.emoji-btn:hover { color: #cbd5e1; }

.upload-btn, .send-btn { 
  width: 48px; height: 48px; border-radius: 50%; border: none; 
  display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
  cursor: pointer; transition: all 0.2s;
}
.upload-btn {
  background: var(--border-color-light); color: #cbd5e1;
}
.upload-btn:hover { background: var(--border-color); }
.send-btn { 
  background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; 
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}
.send-btn:hover { 
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
}

/* 右侧 AI 鉴图区 */
/* 滑动动画 */
.slide-panel-enter-active, .slide-panel-leave-active {
  transition: all 0.4s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.slide-panel-enter-from, .slide-panel-leave-to {
  transform: translateX(100%);
  opacity: 0;
  width: 0;
}
.slide-panel-enter-to, .slide-panel-leave-from {
  transform: translateX(0);
  opacity: 1;
  width: 400px;
}

.ai-panel {
  width: 400px;
  background: var(--panel-bg);
  border-left: 1px solid var(--border-color-light);
  display: flex; flex-direction: column;
  z-index: 20;
  box-shadow: -10px 0 30px rgba(0,0,0,0.2);
}
.ai-header {
  padding: 24px; 
  background: linear-gradient(135deg, rgba(168, 85, 247, 0.12) 0%, rgba(232, 121, 249, 0.06) 100%);
  border-bottom: 1px solid rgba(168, 85, 247, 0.2); 
  display: flex; justify-content: space-between; align-items: center;
}
.title-with-icon {
  display: flex; align-items: center; gap: 12px;
  color: #e879f9; font-weight: 600; font-size: 1.15rem; letter-spacing: 0.5px;
}
.icon-box {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, rgba(232, 121, 249, 0.2), rgba(192, 132, 252, 0.2));
  display: flex; align-items: center; justify-content: center;
  border: 1px solid rgba(232, 121, 249, 0.3);
}

.ai-content { padding: 24px; overflow-y: auto; color: #e2e8f0; flex: 1;}

/* 扫描分析状态 */
.judging-state { padding: 40px 10px; }
.scanning-animation {
  width: 100px; height: 100px; margin: 0 auto 30px;
  border-radius: 50%; border: 2px dashed rgba(168, 85, 247, 0.3);
  position: relative; display: flex; align-items: center; justify-content: center;
}
.ai-core-icon { font-size: 2.5rem; color: #c084fc; animation: pulse 2s infinite; }
.scan-line {
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: #c084fc; box-shadow: 0 0 10px #c084fc;
  animation: scan 2s linear infinite;
}
@keyframes scan {
  0% { top: 0; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

.judging-title { text-align: center; color: #e2e8f0; margin-bottom: 30px; font-weight: 500;}
.judging-steps { list-style: none; padding: 0; margin: 0; }
.step {
  padding: 16px; margin-bottom: 12px; border-radius: 12px;
  background: var(--border-color-light); border: 1px solid var(--border-color-light);
  font-size: 0.9rem; color: var(--text-muted); display: flex; align-items: center; gap: 12px;
  transition: all 0.3s;
}
.step.done { color: #10b981; background: rgba(16, 185, 129, 0.05); border-color: rgba(16, 185, 129, 0.2); }
.step.active { color: #c084fc; background: rgba(192, 132, 252, 0.08); border-color: rgba(192, 132, 252, 0.3); box-shadow: 0 4px 15px rgba(168, 85, 247, 0.1);}
.step.skipped { color: #64748b; opacity: 0.6; }
.step-note { font-size: 0.75rem; color: #94a3b8; margin-left: 4px; }

/* VL 开关 */
.vl-toggle { padding: 12px 0; }
.toggle-label { display: flex; align-items: center; gap: 10px; cursor: pointer; font-size: 0.9rem; color: #cbd5e1; }
.toggle-label input[type="checkbox"] { accent-color: #8b5cf6; width: 16px; height: 16px; }
.toggle-text { display: flex; align-items: center; gap: 6px; }
.toggle-text i { color: #8b5cf6; }

/* 结果状态 */
.fade-in { animation: fadeIn 0.5s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.status-card {
  display: flex; align-items: center; gap: 16px;
  padding: 20px; border-radius: 16px; margin-bottom: 24px;
}
.status-card.fake {
  background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.05));
  border: 1px solid rgba(239, 68, 68, 0.3);
  box-shadow: 0 8px 25px rgba(239, 68, 68, 0.1);
}
.status-card.real {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
  border: 1px solid rgba(16, 185, 129, 0.3);
  box-shadow: 0 8px 25px rgba(16, 185, 129, 0.1);
}
.status-card.real .status-icon { background: rgba(16, 185, 129, 0.2); color: #10b981; }
.status-card.real .status-info h4 { color: #6ee7b7; }
.status-card.real .confidence { color: #34d399; background: rgba(16, 185, 129, 0.1); }
.status-card.error {
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(217, 119, 6, 0.05));
  border: 1px solid rgba(245, 158, 11, 0.3);
  box-shadow: 0 8px 25px rgba(245, 158, 11, 0.1);
}
.status-card.error .status-icon { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
.status-card.error .status-info h4 { color: #fcd34d; }
.status-card.error .confidence { color: #fbbf24; background: rgba(245, 158, 11, 0.1); }
.status-icon {
  width: 48px; height: 48px; border-radius: 12px;
  background: rgba(239, 68, 68, 0.2); color: #ef4444;
  display: flex; align-items: center; justify-content: center; font-size: 1.5rem;
}
.status-info h4 { margin: 0 0 6px 0; color: #fca5a5; font-size: 1.1rem; }
.confidence { font-size: 0.85rem; color: #f87171; background: rgba(239, 68, 68, 0.1); padding: 4px 10px; border-radius: 20px;}

.panel-section { margin-bottom: 24px; }
.section-title {
  font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}
.section-title i { color: #8b5cf6; }

/* 概率分布条 */
.prob-bars { display: flex; flex-direction: column; gap: 10px; }
.prob-item { display: flex; align-items: center; gap: 10px; }
.prob-label { width: 80px; font-size: 0.82rem; color: #94a3b8; text-align: right; flex-shrink: 0; }
.prob-bar-bg { flex: 1; height: 8px; background: var(--border-color-light); border-radius: 4px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #6366f1, #8b5cf6); transition: width 0.8s ease; }
.prob-bar-fill.highlight { background: linear-gradient(90deg, #ef4444, #f87171); }
.prob-value { width: 52px; font-size: 0.82rem; color: #cbd5e1; text-align: left; flex-shrink: 0; }

.heatmap-placeholder {
  position: relative; border-radius: 12px; overflow: hidden;
  border: 1px solid var(--border-color);
}
.heatmap-placeholder img { width: 100%; display: block; }
.heatmap-overlay {
  position: absolute; bottom: 0; left: 0; right: 0;
  background: linear-gradient(to top, rgba(0,0,0,0.8), transparent);
  padding: 30px 16px 12px;
}
.hotspot-label { background: rgba(239, 68, 68, 0.8); color: white; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 500; backdrop-filter: blur(4px);}

.logic-report { display: flex; flex-direction: column; gap: 12px; }
.report-block {
  background: var(--card-bg); border: 1px solid var(--border-color-light);
  padding: 16px; border-radius: 12px; position: relative;
}
.report-tag {
  position: absolute; top: -10px; left: 16px;
  font-size: 0.75rem; font-weight: 600; padding: 2px 10px; border-radius: 12px;
  color: white; box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.report-tag.context { background: linear-gradient(135deg, #3b82f6, #2563eb); }
.report-tag.visual { background: linear-gradient(135deg, #8b5cf6, var(--primary-color)); }
.report-block p { margin: 8px 0 0; font-size: 0.9rem; line-height: 1.6; color: var(--text-main); }
.vl-text.markdown-body { color: var(--text-main); }
.vl-text.markdown-body p, .vl-text.markdown-body li { color: var(--text-main); margin-bottom: 8px; }
.vl-text.markdown-body strong { color: var(--text-main); font-weight: 600; }
.vl-text.markdown-body h1, .vl-text.markdown-body h2, .vl-text.markdown-body h3 { color: var(--text-main); margin: 12px 0 6px; }

/* VL 按钮 & 占位 */
.vl-trigger-btn {
  margin-left: auto; padding: 4px 12px; border-radius: 8px; font-size: 0.8rem; font-weight: 500;
  background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; border: none;
  cursor: pointer; display: flex; align-items: center; gap: 4px; transition: all 0.2s;
}
.vl-trigger-btn:hover { filter: brightness(1.1); box-shadow: 0 2px 10px rgba(139, 92, 246, 0.3); }
.vl-text { white-space: pre-wrap; }
.vl-placeholder { text-align: center; padding: 16px; color: #64748b; font-size: 0.85rem; }
.cursor-blink { animation: blink 1s step-end infinite; }
@keyframes blink { 50% { opacity: 0; } }

.verdict-box {
  background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(217, 119, 6, 0.05));
  border: 1px solid rgba(245, 158, 11, 0.2);
  padding: 20px; border-radius: 16px; text-align: center;
}
.verdict-title { color: #fbbf24; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 1.1rem;}
.verdict-box p { color: #e2e8f0; font-size: 0.95rem; margin-bottom: 16px; line-height: 1.5; }
.verdict-box p strong { color: #ef4444; font-size: 1.1em;}
.apply-verdict-btn {
  width: 100%; background: linear-gradient(135deg, #f59e0b, #d97706); color: white;
  border: none; padding: 12px; border-radius: 10px; font-weight: 600; font-size: 1rem;
  cursor: pointer; transition: all 0.2s; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
}
.apply-verdict-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(245, 158, 11, 0.4); filter: brightness(1.05); }

/* 新建会话输入条 */
.add-session-row {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 20px 12px;
  animation: slideDown 0.2s ease;
}
.join-row { padding-top: 4px; }
.join-hash { color: #60a5fa; font-size: 0.9rem; flex-shrink: 0; }
.join-error { font-size: 0.8rem; color: #f87171; padding: 0 20px 10px; display: flex; align-items: center; gap: 5px; }
@keyframes slideDown { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
.add-session-row input {
  flex: 1; background: var(--border-color-light); border: 1px solid rgba(59, 130, 246, 0.45);
  border-radius: 8px; padding: 8px 12px; color: var(--text-main); font-size: 0.88rem; outline: none;
}
.add-session-row input::placeholder { color: var(--text-muted); }
.add-session-confirm, .add-session-cancel {
  width: 30px; height: 30px; border-radius: 8px; border: none; cursor: pointer;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: all 0.15s;
}
.add-session-confirm { background: rgba(59, 130, 246, 0.2); color: #60a5fa; }
.add-session-confirm:hover { background: rgba(59, 130, 246, 0.4); }
.add-session-cancel { background: var(--border-color-light); color: var(--text-muted); }
.add-session-cancel:hover { background: var(--border-color); }

/* 标题重命名 */
.rename-wrapper { display: flex; align-items: center; gap: 6px; }
.rename-input {
  background: var(--border-color-light); border: 1px solid rgba(99, 102, 241, 0.5);
  border-radius: 8px; padding: 6px 12px; color: var(--text-main); font-size: 0.95rem;
  font-weight: 600; outline: none; width: 200px;
}
.rename-confirm-btn {
  background: rgba(99, 102, 241, 0.15); border: none; color: #818cf8;
  width: 30px; height: 30px; border-radius: 8px; cursor: pointer;
  display: flex; align-items: center; justify-content: center; transition: all 0.15s;
}
.rename-confirm-btn:hover { background: rgba(99, 102, 241, 0.35); }

/* 会话操作下拉菜单 */
.menu-wrapper { position: relative; }
.session-menu {
  position: absolute; right: 0; top: calc(100% + 8px);
  background: var(--panel-bg); border: 1px solid var(--border-color-light);
  border-radius: 10px; padding: 6px; min-width: 150px; z-index: 100;
  box-shadow: 0 8px 25px rgba(0,0,0,0.35);
  animation: fadeIn 0.15s ease;
}
.menu-item {
  display: flex; align-items: center; gap: 8px;
  width: 100%; padding: 9px 12px; border: none; background: none;
  color: var(--text-main); font-size: 0.88rem; border-radius: 7px;
  cursor: pointer; text-align: left; transition: background 0.15s;
}
.menu-item:hover { background: var(--border-color-light); }
.menu-item.danger { color: #f87171; }
.menu-item.danger:hover { background: rgba(239, 68, 68, 0.1); }
.menu-item i { width: 14px; text-align: center; }

/* 邀请码弹窗 */
.invite-mask {
  position: fixed; inset: 0; z-index: 1000;
  background: rgba(0,0,0,0.55);
  display: flex; align-items: center; justify-content: center;
  backdrop-filter: blur(4px);
}
.invite-modal {
  background: var(--panel-bg); border: 1px solid var(--border-color-light);
  border-radius: 20px; padding: 32px; width: min(460px, 90vw);
  box-shadow: 0 24px 60px rgba(0,0,0,0.5);
  animation: fadeIn 0.2s ease;
}
.invite-modal-header {
  display: flex; align-items: center; gap: 12px; margin-bottom: 16px;
}
.invite-modal-header i { color: #818cf8; font-size: 1.3rem; }
.invite-modal-header h3 { margin: 0; flex: 1; color: var(--text-main); font-size: 1.15rem; }
.invite-close {
  background: none; border: none; color: var(--text-muted);
  font-size: 1.1rem; cursor: pointer; padding: 4px; transition: color 0.2s;
}
.invite-close:hover { color: var(--text-main); }
.invite-tip { font-size: 0.88rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 20px; }
.invite-code-box {
  display: flex; align-items: center; gap: 10px;
  background: var(--border-color-light); border: 1px solid var(--border-color);
  border-radius: 12px; padding: 12px 16px; margin-bottom: 20px;
}
.invite-code {
  flex: 1; font-family: monospace; font-size: 0.82rem; word-break: break-all;
  color: #a5b4fc; line-height: 1.5;
}
.copy-btn {
  flex-shrink: 0; background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.3);
  color: #818cf8; padding: 7px 14px; border-radius: 8px; font-size: 0.85rem;
  cursor: pointer; display: flex; align-items: center; gap: 5px; transition: all 0.2s;
}
.copy-btn:hover { background: rgba(99,102,241,0.3); }
.invite-ok-btn {
  width: 100%; background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: white; border: none; padding: 12px; border-radius: 10px;
  font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: all 0.2s;
}
.invite-ok-btn:hover { filter: brightness(1.08); }
</style>

