<script setup lang="ts">
import { ref } from 'vue';

// 模拟前端 UI 数据以供演示排版，后续从后端接口拉取
const sessions = ref([
  { id: 1, title: '关于"全新显卡"的真伪纠纷', status: 'ai_judging', time: '10:01' },
  { id: 2, title: '限量球鞋实物图是否由AI生成？', status: 'open', time: '昨天' }
]);

const activeSession = ref(sessions.value[0]);

const messages = ref([
  { id: 1, role: 'seller', name: '卖家', content: '兄弟，这是我新拍的高端显卡实物图，绝对保真！', type: 'text' },
  { id: 2, role: 'seller', name: '卖家', content: 'https://via.placeholder.com/300x200?text=GPU+Image', type: 'image', hasBeenDetected: false },
  { id: 3, role: 'buyer', name: '买家(我)', content: '别扯了，这图背景阴影都不对，边缘变形了，肯定是 AI 跑出来的！', type: 'text' },
  { id: 4, role: 'seller', name: '卖家', content: '胡说八道，我不懂什么AI，你爱买不买！', type: 'text' },
  { id: 5, role: 'buyer', name: '买家(我)', content: '@AI法官 帮我检测一下上面那张图！', type: 'text' },
  { id: 6, role: 'system', name: '系统', content: '买家请求 AI 介入检测，正在调取图片特征...', type: 'text' }
]);

const showAiPanel = ref(false);
const isJudging = ref(false);

const requestAiDetection = (msgId: number) => {
  showAiPanel.value = true;
  isJudging.value = true;
  
  // 模拟 AI 处理两秒后出结果
  setTimeout(() => {
    isJudging.value = false;
  }, 2000);
};

</script>

<template>
  <div class="chat-container">
    
    <!-- 左侧：纠纷会话列表 -->
    <div class="sidebar">
      <div class="sidebar-header">
        <h3>纠纷法庭中心</h3>
        <button class="add-btn"><i class="fa-solid fa-plus"></i></button>
      </div>
      <ul class="session-list">
        <li 
          v-for="item in sessions" 
          :key="item.id" 
          :class="['session-item', { active: activeSession.id === item.id }]"
          @click="activeSession = item"
        >
          <div class="session-info">
            <span class="session-title">{{ item.title }}</span>
            <span class="session-time">{{ item.time }}</span>
          </div>
          <div class="session-status">
            <span v-if="item.status === 'ai_judging'" class="badge judging"><i class="fa-solid fa-robot"></i> AI介入中</span>
            <span v-if="item.status === 'open'" class="badge open">未解决</span>
          </div>
        </li>
      </ul>
    </div>

    <!-- 中间：聊天/辩论流 -->
    <div class="chat-main">
      <div class="chat-header">
        <h3>{{ activeSession.title }}</h3>
        <span class="status-text">当前状态: 申诉中</span>
      </div>
      
      <div class="message-list">
        <div 
          v-for="msg in messages" 
          :key="msg.id" 
          :class="['message-wrapper', msg.role]"
        >
          <div v-if="msg.role === 'system'" class="system-message">
            <span>{{ msg.content }}</span>
          </div>
          
          <template v-else>
            <div class="avatar">{{ msg.name[0] }}</div>
            <div class="message-content-wrapper">
              <div class="sender-name">{{ msg.name }}</div>
              
              <!-- 文本消息 -->
              <div class="message-bubble" v-if="msg.type === 'text'">
                {{ msg.content }}
              </div>

              <!-- 图片消息 -->
              <div class="image-bubble" v-else-if="msg.type === 'image'">
                <img :src="msg.content" alt="纠纷证据图" />
                <div class="image-actions">
                  <button class="detect-btn" @click="requestAiDetection(msg.id)">
                    <i class="fa-solid fa-microscope"></i> AI 入场鉴定
                  </button>
                </div>
              </div>
            </div>
          </template>
        </div>
      </div>

      <div class="chat-input-area">
        <button class="upload-btn"><i class="fa-solid fa-image"></i></button>
        <input type="text" placeholder="输入反驳证据或请求 @AI法官 协助..." />
        <button class="send-btn"><i class="fa-solid fa-paper-plane"></i></button>
      </div>
    </div>

    <!-- 右侧：AI 裁决面板 (条件渲染) -->
    <div class="ai-panel" v-if="showAiPanel">
      <div class="ai-header">
        <i class="fa-solid fa-gavel"></i> AI 鉴定法庭
        <button class="close-btn" @click="showAiPanel = false"><i class="fa-solid fa-xmark"></i></button>
      </div>
      
      <div class="ai-content">
        <div v-if="isJudging" class="judging-state">
           <div class="spinner"></div>
           <p>AI 正在通过 LaRE 模型提取图像频域特征...</p>
           <p>Qwen-VL 正在审阅聊天上下文...</p>
        </div>
        <div v-else class="result-state">
            <div class="status-badge fake">
              <i class="fa-solid fa-robot"></i> 判定结果: AI生成 (证据确凿)
            </div>
            
            <div class="heatmap-placeholder">
              <!-- 这里后续可以直接复用你之前的 HeatmapSlider 组件 -->
              <span>[热力图展示区]</span>
            </div>

            <div class="logic-report">
              <h4>💬 裁判长多模态报告：</h4>
              <p>结合买卖双方对话上下文：“买家指出的边缘变形问题属实”。</p>
              <p>从物理层面的 AI 检测来看，图像左上角和背景光影存在严重的频域伪影，模型置信度高达 98.7%。属于典型的 AIGC 生成图像。</p>
              <p><strong>判决建议</strong>：支持买家诉求，卖家存在虚假商品展示行为，退款处理。</p>
            </div>
        </div>
      </div>
    </div>

  </div>
</template>

<style scoped>
.chat-container {
  display: flex;
  height: calc(100vh - 100px); /* 减去顶部导航栏高度 */
  width: 100%;
  max-width: 1400px;
  margin: 80px auto 20px;
  background: rgba(15, 23, 42, 0.4);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}

/* 左侧列表 */
.sidebar {
  width: 280px;
  background: rgba(0, 0, 0, 0.2);
  border-right: 1px solid rgba(255, 255, 255, 0.05);
  display: flex;
  flex-direction: column;
}
.sidebar-header {
  padding: 20px;
  color: #e2e8f0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.add-btn {
  background: rgba(255,255,255,0.1);
  border: none;
  color: white;
  width: 30px; height: 30px; border-radius: 6px; cursor: pointer;
}
.session-list {
  list-style: none; padding: 0; margin: 0; overflow-y: auto;
}
.session-item {
  padding: 15px 20px;
  border-bottom: 1px solid rgba(255,255,255,0.02);
  cursor: pointer;
  transition: background 0.2s;
}
.session-item:hover { background: rgba(255,255,255,0.03); }
.session-item.active { background: rgba(96, 165, 250, 0.1); border-left: 3px solid #60a5fa; }
.session-title { color: #e2e8f0; font-size: 0.95em; font-weight: 500; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.session-time { color: #64748b; font-size: 0.75em; }
.badge { font-size: 0.7em; padding: 2px 6px; border-radius: 4px; margin-top: 5px; display: inline-block;}
.badge.judging { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
.badge.open { background: rgba(148, 163, 184, 0.2); color: #94a3b8; }

/* 中间聊天区 */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
}
.chat-header {
  padding: 20px;
  border-bottom: 1px solid rgba(255,255,255,0.05);
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #f1f5f9;
}
.status-text { font-size: 0.85em; color: #94a3b8; }

.message-list {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 15px;
}
.message-wrapper { display: flex; gap: 12px; max-width: 80%; }
.message-wrapper.buyer { align-self: flex-end; flex-direction: row-reverse; }
.message-wrapper.seller { align-self: flex-start; }

.system-message { align-self: center; margin: 10px 0; max-width: 100%;}
.system-message span { background: rgba(255,255,255,0.05); color: #94a3b8; padding: 5px 12px; border-radius: 12px; font-size: 0.8em; }

.avatar { width: 36px; height: 36px; border-radius: 50%; background: #475569; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;}
.buyer .avatar { background: #3b82f6; }
.seller .avatar { background: #f59e0b; }

.message-content-wrapper { display: flex; flex-direction: column; }
.buyer .message-content-wrapper { align-items: flex-end; }
.sender-name { font-size: 0.75em; color: #64748b; margin-bottom: 4px; }

.message-bubble {
  background: rgba(255,255,255,0.05);
  color: #e2e8f0;
  padding: 10px 15px;
  border-radius: 2px 12px 12px 12px;
  line-height: 1.5;
  font-size: 0.95em;
  word-break: break-word;
}
.buyer .message-bubble { background: #2563eb; color: white; border-radius: 12px 2px 12px 12px; }

.image-bubble {
  background: rgba(0,0,0,0.2);
  padding: 8px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.05);
}
.image-bubble img { max-width: 250px; border-radius: 8px; display: block; }
.image-actions { margin-top: 8px; text-align: center; }
.detect-btn {
  background: linear-gradient(135deg, #8b5cf6, #6366f1);
  color: white; border: none; padding: 6px 15px; border-radius: 6px; font-size: 0.85em; cursor: pointer;
  transition: all 0.2s; box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
}
.detect-btn:hover { filter: brightness(1.1); transform: translateY(-1px); }

.chat-input-area {
  padding: 20px;
  background: rgba(0,0,0,0.2);
  border-top: 1px solid rgba(255,255,255,0.05);
  display: flex; gap: 10px;
}
.chat-input-area input {
  flex: 1;
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
  color: white; border-radius: 8px; padding: 0 15px; outline: none;
}
.chat-input-area input::placeholder { color: #64748b; }
.upload-btn, .send-btn { 
  width: 45px; height: 45px; border-radius: 8px; border: none; 
  background: rgba(255,255,255,0.05); color: #cbd5e1; cursor: pointer; transition: 0.2s;
}
.send-btn { background: #3b82f6; color: white; }
.send-btn:hover { background: #2563eb; }

/* 右侧 AI 鉴图区 */
.ai-panel {
  width: 380px;
  background: rgba(0, 0, 0, 0.3);
  border-left: 1px solid rgba(255, 255, 255, 0.05);
  display: flex; flex-direction: column;
}
.ai-header {
  padding: 20px; color: #c084fc; font-weight: bold; font-size: 1.1em;
  border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center;
}
.close-btn { background: none; border: none; color: #94a3b8; cursor: pointer; font-size: 1.2em; }
.ai-content { padding: 20px; overflow-y: auto; color: #e2e8f0; }
.judging-state { text-align: center; color: #94a3b8; padding: 40px 0; line-height: 1.8; }
.judging-state .spinner { margin: 0 auto 20px; }

.status-badge.fake {
  background: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.2);
  padding: 12px; border-radius: 8px; font-weight: bold; text-align: center; margin-bottom: 20px;
}
.heatmap-placeholder {
  height: 200px; background: rgba(255,255,255,0.05); border: 1px dashed rgba(255,255,255,0.1);
  border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #64748b; margin-bottom: 20px;
}
.logic-report p { font-size: 0.9em; line-height: 1.6; color: #cbd5e1; margin-bottom: 10px; }
.logic-report h4 { margin-bottom: 10px; color: #f8fafc; }
</style>

