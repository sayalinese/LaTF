import axios from 'axios';
import type { PredictResponse, ConfigResponse, ExplainLogicResponse } from '../types/api';

const apiClient = axios.create({
  baseURL: '/api', // Vite 代理指向 /api -> http://127.0.0.1:5000
  timeout: 60000,
});

export const fetchConfig = async (): Promise<ConfigResponse> => {
  const { data } = await apiClient.get<ConfigResponse>('/config');
  return data;
};

export const predictImage = async (file: File): Promise<PredictResponse> => {
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