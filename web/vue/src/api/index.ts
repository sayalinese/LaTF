import axios from 'axios';
import type { PredictResponse, ConfigResponse, ExplainLogicResponse } from '../types/api';

const apiClient = axios.create({
  baseURL: '/api', // Vite 代理指向 /api -> http://127.0.0.1:5555
  timeout: 60000,
  withCredentials: true,
});

const USE_MOCK = false;

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export const fetchConfig = async (): Promise<ConfigResponse> => {
  if (USE_MOCK) {
    await delay(500);
    return {
      model_version: "LaRE-v11",
      cascade_enabled: true,
      cascade_threshold: 0.9,
      ai_confidence_threshold: 0.5,
      lare_model_type: "fusion",
      device: "cuda:0",
      dual_detector_enabled: true,
      resolved_out_dir: "./outputs",
      ckpt_path: "mock_path.pth"
    };
  }
  const { data } = await apiClient.get<ConfigResponse>('/config');
  return data;
};

export const predictImage = async (file: File): Promise<PredictResponse> => {
  if (USE_MOCK) {
    await delay(1500);
    const fakeProb = Math.random();
    const isFake = fakeProb > 0.5;
    return {
      class_name: isFake ? "AIGC/Fake" : "Real",
      confidence: fakeProb,
      class_idx: isFake ? 1 : 0,
      probabilities: {
        "AIGC/Fake": fakeProb,
        "Real": 1 - fakeProb
      },
      heatmap: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
      debug: {
        mock: true,
        execution_time: 1.23,
        model_used: "model_v11_fusion"
      }
    };
  }
  const formData = new FormData();
  formData.append('file', file);
  
  const { data } = await apiClient.post<PredictResponse>('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return data;
};

export const explainLogic = async (base64Image: string, isFake: boolean): Promise<ExplainLogicResponse> => {
  if (USE_MOCK) {
    await delay(2000);
    return {
      report: "这是一段根据Mock数据生成的分析解释报告。图中显示的特征证明了这是AIGC生成的..."
    };
  }
  const { data } = await apiClient.post<ExplainLogicResponse>('/explain_logic', {
    image: base64Image,
    is_fake: isFake
  }, {
    timeout: 180000
  });
  return data;
};

export const explainLogicStream = async (
  base64Image: string, 
  isFake: boolean,
  onChunk: (chunk: string) => void
): Promise<void> => {
  if (USE_MOCK) {
    const text = "这是一段流式输出的Mock逻辑分析报告内容。可以看到前端页面能够一字一句地显示出来的效果...";
    for (let i = 0; i < text.length; i++) {
      await delay(50);
      onChunk(text[i]);
    }
    return;
  }
  const response = await fetch('/api/explain_logic_stream', { // Vite proxy adds /api prefix
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      image: base64Image,
      is_fake: isFake
    })
  });

  if (!response.ok) {
    throw new Error(`流式请求失败: ${response.status} ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error("无法读取流数据");

  const decoder = new TextDecoder('utf-8');
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      onChunk(decoder.decode(value, { stream: true }));
    }
  }
};
export const login = async (data: any) => {
  const res = await apiClient.post('/login', data);
  return res.data;
};

export const register = async (data: any) => {
  const res = await apiClient.post('/register', data);
  return res.data;
};

export const logout = async () => {
  const res = await apiClient.post('/logout');
  return res.data;
};

export const fetchCurrentUser = async () => {
  const res = await apiClient.get('/me');
  return res.data;
};

export const updateProfile = async (data: { nickname?: string; avatar?: File }) => {
  const formData = new FormData();
  if (data.nickname) formData.append('nickname', data.nickname);
  if (data.avatar) formData.append('avatar', data.avatar);
  const res = await apiClient.post('/update_profile', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
};

export const fetchSessions = async () => {
  const res = await apiClient.get('/sessions');
  return res.data;
};

export const fetchSessionMessages = async (sessionId: string, after?: string) => {
  const url = after
    ? `/sessions/${sessionId}/messages?after=${encodeURIComponent(after)}`
    : `/sessions/${sessionId}/messages`;
  const res = await apiClient.get(url);
  return res.data;
};

export const sendTextMessage = async (sessionId: string, content: string, senderRole = 'buyer') => {
  const formData = new FormData();
  formData.append('content', content);
  formData.append('content_type', 'text');
  formData.append('sender_role', senderRole);
  const res = await apiClient.post(`/sessions/${sessionId}/messages`, formData);
  return res.data;
};

export const sendImageMessage = async (sessionId: string, file: File, senderRole = 'buyer') => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('content_type', 'image');
  formData.append('sender_role', senderRole);
  const res = await apiClient.post(`/sessions/${sessionId}/messages`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
};

export const detectMessage = async (messageId: string, useVl = false) => {
  const res = await apiClient.post('/detect_message', { message_id: messageId, use_vl: useVl }, { timeout: 120000 });
  return res.data;
};

export const createSession = async (topicName: string, platform: string = 'taobao', orderInfo?: { order_id?: string; product_name?: string; product_price?: string; product_image?: string }) => {
  const res = await apiClient.post('/sessions', { topic_name: topicName, platform, ...orderInfo });
  return res.data;
};

export const updateSession = async (sessionId: string, data: Record<string, any>) => {
  const res = await apiClient.put(`/sessions/${sessionId}`, data);
  return res.data;
};

export const changePlatform = async (sessionId: string, platform: string) => {
  const res = await apiClient.put(`/sessions/${sessionId}`, { platform });
  return res.data;
};

export const deleteSession = async (sessionId: string) => {
  const res = await apiClient.delete(`/sessions/${sessionId}`);
  return res.data;
};

export const joinSession = async (sessionId: string) => {
  const res = await apiClient.post(`/sessions/${sessionId}/join`);
  return res.data;
};

export const assistantChatStream = async (
  sessionId: string,
  message: string,
  history: Array<{ role: string; content: string }>,
  onChunk: (chunk: string) => void,
  platform?: string
): Promise<void> => {
  const response = await fetch(`/api/sessions/${sessionId}/assistant_chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ message, history, platform })
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ message: '请求失败' }));
    throw new Error(err.message || `请求失败: ${response.status}`);
  }
  const reader = response.body?.getReader();
  if (!reader) throw new Error("无法读取流数据");
  const decoder = new TextDecoder('utf-8');
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) onChunk(decoder.decode(value, { stream: true }));
  }
};
