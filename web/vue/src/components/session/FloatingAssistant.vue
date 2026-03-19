<script setup lang="ts">
import { ref, watch, nextTick, computed, onMounted, onUnmounted } from 'vue';
import { assistantChatStream } from '../../api';
import { marked } from 'marked';

// marked 配置
marked.setOptions({ breaks: true, gfm: true });

const renderMd = (text: string): string => {
  try {
    return marked.parse(text) as string;
  } catch {
    return text;
  }
};

const props = defineProps<{
  sessionId: string | null;
  sessionTitle: string;
}>();

const isOpen = ref(false);
const inputText = ref('');
const isStreaming = ref(false);
const chatListEl = ref<HTMLElement | null>(null);

interface ChatMsg {
  role: 'user' | 'assistant';
  content: string;
}

const chatHistory = ref<ChatMsg[]>([]);

// 会话切换时清空聊天
watch(() => props.sessionId, () => {
  chatHistory.value = [];
  inputText.value = '';
  isStreaming.value = false;
});

const disabled = computed(() => !props.sessionId);

// ── 拖拽逻辑 ──────────────────────────────────
const FAB_SIZE = 56;
const PANEL_W = 380;
const PANEL_H = 520;

const pos = ref({ x: 0, y: 0 });
const isDragging = ref(false);
const hasDragged = ref(false);
const dragStart = ref({ mx: 0, my: 0, px: 0, py: 0 });

const clampPos = (x: number, y: number) => ({
  x: Math.max(0, Math.min(window.innerWidth - FAB_SIZE, x)),
  y: Math.max(0, Math.min(window.innerHeight - FAB_SIZE, y)),
});

const fabStyle = computed(() => ({
  position: 'fixed' as const,
  left: pos.value.x + 'px',
  top: pos.value.y + 'px',
  cursor: isDragging.value ? 'grabbing' : 'grab',
}));

const panelStyle = computed(() => {
  const margin = 12;
  let top = pos.value.y - PANEL_H - margin;
  if (top < margin) top = pos.value.y + FAB_SIZE + margin;
  let left = pos.value.x + FAB_SIZE - PANEL_W;
  left = Math.max(margin, Math.min(window.innerWidth - PANEL_W - margin, left));
  top  = Math.max(margin, Math.min(window.innerHeight - PANEL_H - margin, top));
  return { position: 'fixed' as const, left: left + 'px', top: top + 'px' };
});

// 鼠标拖拽
const onDragStart = (e: MouseEvent) => {
  isDragging.value = true;
  hasDragged.value = false;
  dragStart.value = { mx: e.clientX, my: e.clientY, px: pos.value.x, py: pos.value.y };
  e.preventDefault();
};

const onMouseMove = (e: MouseEvent) => {
  if (!isDragging.value) return;
  const dx = e.clientX - dragStart.value.mx;
  const dy = e.clientY - dragStart.value.my;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) hasDragged.value = true;
  pos.value = clampPos(dragStart.value.px + dx, dragStart.value.py + dy);
};

const onMouseUp = () => {
  if (!isDragging.value) return;
  isDragging.value = false;
  if (!hasDragged.value) togglePanel();
};

// 触控拖拽
const onTouchStart = (e: TouchEvent) => {
  if (disabled.value) return;
  const t = e.touches[0];
  isDragging.value = true;
  hasDragged.value = false;
  dragStart.value = { mx: t.clientX, my: t.clientY, px: pos.value.x, py: pos.value.y };
};

const onTouchMove = (e: TouchEvent) => {
  if (!isDragging.value) return;
  e.preventDefault();
  const t = e.touches[0];
  const dx = t.clientX - dragStart.value.mx;
  const dy = t.clientY - dragStart.value.my;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) hasDragged.value = true;
  pos.value = clampPos(dragStart.value.px + dx, dragStart.value.py + dy);
};

const onTouchEnd = () => {
  if (!isDragging.value) return;
  isDragging.value = false;
  if (!hasDragged.value) togglePanel();
};

onMounted(() => {
  pos.value = clampPos(window.innerWidth - FAB_SIZE - 32, window.innerHeight - FAB_SIZE - 32);
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mouseup', onMouseUp);
  document.addEventListener('touchmove', onTouchMove, { passive: false });
  document.addEventListener('touchend', onTouchEnd);
});

onUnmounted(() => {
  document.removeEventListener('mousemove', onMouseMove);
  document.removeEventListener('mouseup', onMouseUp);
  document.removeEventListener('touchmove', onTouchMove);
  document.removeEventListener('touchend', onTouchEnd);
});

const scrollToBottom = async () => {
  await nextTick();
  if (chatListEl.value) chatListEl.value.scrollTop = chatListEl.value.scrollHeight;
};

const togglePanel = () => {
  if (disabled.value) return;
  isOpen.value = !isOpen.value;
  if (isOpen.value) scrollToBottom();
};

const quickSend = (text: string) => {
  if (isStreaming.value) return;
  inputText.value = text;
  sendMessage();
};

const sendMessage = async () => {
  const text = inputText.value.trim();
  if (!text || isStreaming.value || !props.sessionId) return;

  chatHistory.value.push({ role: 'user', content: text });
  inputText.value = '';
  scrollToBottom();

  // 添加占位助手消息
  chatHistory.value.push({ role: 'assistant', content: '' });
  const assistantIdx = chatHistory.value.length - 1;
  isStreaming.value = true;

  try {
    // 传历史（不含当前空消息）
    const historyForApi = chatHistory.value.slice(0, -1).map(m => ({
      role: m.role,
      content: m.content
    }));

    await assistantChatStream(
      props.sessionId,
      text,
      historyForApi,
      (chunk: string) => {
        chatHistory.value[assistantIdx].content += chunk;
        scrollToBottom();
      }
    );
  } catch (e: any) {
    chatHistory.value[assistantIdx].content += `\n[请求失败: ${e.message}]`;
  } finally {
    isStreaming.value = false;
    scrollToBottom();
  }
};
</script>

<template>
  <teleport to="body">
    <!-- 悬浮按钮 -->
    <button
      class="fab"
      :style="fabStyle"
      :class="{ 'fab-active': isOpen, 'fab-disabled': disabled, 'fab-dragging': isDragging }"
      :title="disabled ? '请先选择一个会话' : (isOpen ? '收起助手' : 'AI 小助手')"
      @mousedown="onDragStart"
      @touchstart.prevent="onTouchStart"
    >
      <svg v-if="!isOpen" class="fab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="8" y1="9" x2="8" y2="9.01"></line>
        <line x1="16" y1="9" x2="16" y2="9.01"></line>
        <path d="M9 16c.5 1 1.4 2 3 2s2.5-1 3-2"></path>
      </svg>
      <svg v-else class="fab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3">
        <line x1="18" y1="6" x2="6" y2="18"></line>
        <line x1="6" y1="6" x2="18" y2="18"></line>
      </svg>
    </button>

    <!-- 聊天面板 -->
    <transition name="assistant-pop">
      <div v-if="isOpen" class="assistant-panel" :style="panelStyle">
        <div class="assistant-header">
          <div class="header-info">
            <div class="header-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:18px;height:18px">
                <circle cx="12" cy="12" r="10"/>
                <line x1="8" y1="9" x2="8" y2="9.01"/>
                <line x1="16" y1="9" x2="16" y2="9.01"/>
                <path d="M9 16c.5 1 1.4 2 3 2s2.5-1 3-2"/>
              </svg>
            </div>
            <div>
              <div class="header-title">AI 纠纷助手</div>
              <div class="header-sub" :title="sessionTitle">{{ sessionTitle || '当前会话' }}</div>
            </div>
          </div>
          <button class="close-btn" @click="isOpen = false" title="收起">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" style="width:14px;height:14px">
              <line x1="18" y1="6" x2="6" y2="18"/>
              <line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>

        <div class="assistant-chat" ref="chatListEl">
          <!-- 欢迎消息 -->
          <div v-if="chatHistory.length === 0" class="welcome-msg">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class="welcome-icon">
              <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z"/>
            </svg>
            <p>你好！我是 AI 纠纷助手。</p>
            <p class="welcome-sub">我会根据当前聊天记录帮你分析对方行为、提供维权建议。直接问我吧！</p>
            <div class="quick-actions">
              <button class="quick-btn" @click="quickSend('分析对方是否有可疑行为')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="quick-icon"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                分析可疑行为
              </button>
              <button class="quick-btn" @click="quickSend('帮我整理维权要点')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="quick-icon"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><polyline points="3 6 4 7 6 5"/><polyline points="3 12 4 13 6 11"/><polyline points="3 18 4 19 6 17"/></svg>
                整理维权要点
              </button>
              <button class="quick-btn" @click="quickSend('总结目前的聊天情况')">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="quick-icon"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                总结聊天
              </button>
            </div>
          </div>

          <div v-for="(msg, idx) in chatHistory" :key="idx" :class="['chat-msg', msg.role]">
            <div class="msg-avatar" :title="msg.role === 'user' ? '你' : 'AI 助手'">
              <svg v-if="msg.role === 'user'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px;height:14px">
                <circle cx="12" cy="8" r="4"/>
                <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7"/>
              </svg>
              <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px;height:14px">
                <circle cx="12" cy="12" r="10"/>
                <line x1="8" y1="9" x2="8" y2="9.01"/>
                <line x1="16" y1="9" x2="16" y2="9.01"/>
                <path d="M9 16c.5 1 1.4 2 3 2s2.5-1 3-2"/>
              </svg>
            </div>
            <div v-if="msg.role === 'assistant'" class="msg-bubble md-bubble">
              <span v-html="renderMd(msg.content)"></span
              ><span
                v-if="isStreaming && idx === chatHistory.length - 1"
                class="cursor-blink"
              >|</span>
            </div>
            <div v-else class="msg-bubble">{{ msg.content }}</div>
          </div>
        </div>

        <div class="assistant-input">
          <input
            type="text"
            v-model="inputText"
            placeholder="向 AI 助手提问..."
            @keyup.enter="sendMessage"
            :disabled="isStreaming"
          />
          <button class="send-btn" @click="sendMessage" :disabled="isStreaming || !inputText.trim()">
            <svg v-if="!isStreaming" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
            <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:16px;height:16px" class="spin-anim"><path d="M21 12a9 9 0 1 1-6.22-8.56"/></svg>
          </button>
        </div>
      </div>
    </transition>
  </teleport>
</template>

<style>
/* ── FAB ─────────────────────────────────────── */
.fab {
  position: fixed;
  width: 56px;
  height: 56px;
  border-radius: 50%;
  border: none;
  background: linear-gradient(135deg, #8b5cf6, #6366f1);
  color: white;
  font-size: 1.35rem;
  box-shadow: 0 6px 24px rgba(99, 102, 241, 0.45);
  transition: background 0.25s, box-shadow 0.25s, transform 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  user-select: none;
  touch-action: none;
}
.fab:hover:not(.fab-disabled):not(.fab-dragging) {
  transform: scale(1.08);
  box-shadow: 0 10px 32px rgba(99, 102, 241, 0.55);
}
.fab-active {
  background: linear-gradient(135deg, #475569, #334155) !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
}
.fab-disabled {
  background: #334155 !important;
  color: #64748b !important;
  cursor: not-allowed !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}
.fab-dragging {
  transform: scale(1.12);
  transition: none;
}
.fab-icon {
  width: 1.35rem;
  height: 1.35rem;
  display: inline-block;
  color: white;
}

/* ── 面板 ─────────────────────────────────────── */
.assistant-panel {
  position: fixed;
  width: 380px;
  height: 520px;
  background: var(--panel-bg, #0f172a);
  border: 1px solid var(--border-color-light, rgba(255,255,255,0.08));
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255,255,255,0.05) inset;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  z-index: 9998;
}

.assistant-header {
  padding: 14px 16px 14px 20px;
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(99, 102, 241, 0.08) 100%);
  border-bottom: 1px solid rgba(139, 92, 246, 0.2);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}
.close-btn {
  background: none;
  border: none;
  color: #64748b;
  cursor: pointer;
  font-size: 1rem;
  padding: 4px 6px;
  border-radius: 6px;
  transition: color 0.15s, background 0.15s;
  line-height: 1;
  flex-shrink: 0;
}
.close-btn:hover { color: #e2e8f0; background: rgba(255,255,255,0.06); }
.header-info {
  display: flex;
  align-items: center;
  gap: 12px;
}
.header-icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.25), rgba(99, 102, 241, 0.25));
  border: 1px solid rgba(139, 92, 246, 0.35);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #a78bfa;
  font-size: 1.1rem;
}
.header-title {
  color: #e2e8f0;
  font-weight: 600;
  font-size: 0.95rem;
}
.header-sub {
  color: #94a3b8;
  font-size: 0.78rem;
  margin-top: 2px;
  max-width: 240px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* 聊天区 */
.assistant-chat {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.assistant-chat::-webkit-scrollbar { width: 5px; }
.assistant-chat::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

.welcome-msg {
  text-align: center;
  padding: 30px 10px 10px;
  color: #94a3b8;
}
.welcome-msg > i {
  font-size: 2rem;
  color: #8b5cf6;
  margin-bottom: 12px;
  display: block;
}
.welcome-icon {
  width: 2rem;
  height: 2rem;
  color: #8b5cf6;
  margin: 0 auto 12px;
  display: block;
}
.welcome-msg p {
  margin: 0 0 4px;
  color: #cbd5e1;
  font-size: 0.95rem;
}
.welcome-sub {
  font-size: 0.82rem !important;
  color: #64748b !important;
  line-height: 1.5;
}
.quick-actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 18px;
}
.quick-btn {
  background: var(--border-color-light, rgba(255,255,255,0.06));
  border: 1px solid var(--border-color, rgba(255,255,255,0.1));
  color: #a5b4fc;
  padding: 10px 14px;
  border-radius: 10px;
  font-size: 0.85rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s;
  text-align: left;
}
.quick-btn:hover {
  background: rgba(139, 92, 246, 0.1);
  border-color: rgba(139, 92, 246, 0.3);
}
.quick-btn i {
  color: #8b5cf6;
  width: 16px;
  text-align: center;
}
.quick-icon {
  width: 14px;
  height: 14px;
  color: #8b5cf6;
  flex-shrink: 0;
}

/* 消息 */
.chat-msg {
  display: flex;
  gap: 10px;
  max-width: 92%;
}
.chat-msg.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}
.msg-avatar {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  flex-shrink: 0;
}
.chat-msg.assistant .msg-avatar {
  background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(99, 102, 241, 0.2));
  color: #a78bfa;
  border: 1px solid rgba(139, 92, 246, 0.3);
}
.chat-msg.user .msg-avatar {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(37, 99, 235, 0.2));
  color: #60a5fa;
  border: 1px solid rgba(59, 130, 246, 0.3);
}
.msg-bubble {
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 0.88rem;
  line-height: 1.6;
  word-break: break-word;
  white-space: pre-wrap;
}
/* md-bubble 不能用 pre-wrap，让 marked 生成的 HTML 正常流式排版 */
.md-bubble {
  white-space: normal;
}
.chat-msg.assistant .msg-bubble {
  background: var(--card-bg, rgba(255,255,255,0.04));
  border: 1px solid var(--border-color-light, rgba(255,255,255,0.08));
  color: #cbd5e1;
  border-radius: 4px 12px 12px 12px;
}
.chat-msg.user .msg-bubble {
  background: linear-gradient(135deg, #6366f1, #4f46e5);
  color: white;
  border-radius: 12px 4px 12px 12px;
}

.cursor-blink {
  animation: blink 1s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }
.spin-anim { animation: spin-svg 1s linear infinite; }
@keyframes spin-svg { to { transform: rotate(360deg); } }

/* 输入区 */
.assistant-input {
  padding: 14px 16px;
  border-top: 1px solid var(--border-color-light, rgba(255,255,255,0.08));
  display: flex;
  gap: 10px;
  align-items: center;
  background: var(--panel-bg, #0f172a);
}
.assistant-input input {
  flex: 1;
  height: 40px;
  background: var(--border-color-light, rgba(255,255,255,0.06));
  border: 1px solid var(--border-color, rgba(255,255,255,0.1));
  color: #e2e8f0;
  border-radius: 20px;
  padding: 0 16px;
  font-size: 0.88rem;
  outline: none;
  transition: all 0.2s;
}
.assistant-input input:focus {
  border-color: rgba(139, 92, 246, 0.5);
  box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.15);
}
.assistant-input input::placeholder { color: #475569; }
.assistant-input input:disabled { opacity: 0.6; }
.assistant-input .send-btn {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  border: none;
  background: linear-gradient(135deg, #8b5cf6, #6366f1);
  color: white;
  font-size: 0.9rem;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  box-shadow: 0 3px 10px rgba(99, 102, 241, 0.3);
}
.assistant-input .send-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 5px 16px rgba(99, 102, 241, 0.4);
}
.assistant-input .send-btn:disabled {
  background: #334155;
  color: #64748b;
  cursor: not-allowed;
  box-shadow: none;
}

/* 弹出动画 */
.assistant-pop-enter-active {
  transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.assistant-pop-leave-active {
  transition: all 0.2s ease-in;
}
.assistant-pop-enter-from,
.assistant-pop-leave-to {
  opacity: 0;
  transform: scale(0.88) translateY(10px);
}

/* ── Markdown 渲染样式 ────────────────────────── */
.md-bubble p { margin: 0 0 6px; }
.md-bubble p:last-child { margin-bottom: 0; }
.md-bubble h1, .md-bubble h2, .md-bubble h3, .md-bubble h4 {
  color: #e2e8f0;
  margin: 10px 0 6px;
  font-weight: 600;
  line-height: 1.4;
}
.md-bubble h3 { font-size: 0.92rem; color: #a78bfa; }
.md-bubble h4 { font-size: 0.88rem; color: #94a3b8; }
.md-bubble ul, .md-bubble ol {
  padding-left: 18px;
  margin: 4px 0 6px;
}
.md-bubble li { margin-bottom: 3px; }
.md-bubble strong { color: #e2e8f0; font-weight: 600; }
.md-bubble em { color: #a5b4fc; }
.md-bubble code {
  background: rgba(139, 92, 246, 0.15);
  color: #c4b5fd;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 0.82rem;
  font-family: 'Fira Code', monospace;
}
.md-bubble pre {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(139, 92, 246, 0.2);
  border-radius: 8px;
  padding: 10px 12px;
  overflow-x: auto;
  margin: 6px 0;
}
.md-bubble pre code { background: none; padding: 0; color: #cbd5e1; }
.md-bubble blockquote {
  border-left: 3px solid rgba(139, 92, 246, 0.5);
  margin: 6px 0;
  padding: 4px 10px;
  color: #94a3b8;
  font-style: italic;
}
.md-bubble hr {
  border: none;
  border-top: 1px solid rgba(255,255,255,0.1);
  margin: 8px 0;
}
</style>
