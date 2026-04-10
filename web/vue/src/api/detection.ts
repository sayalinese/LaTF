/**
 * 统一的检测 API 模块
 * 集中管理所有图像检测相关的 API 调用
 */
import axios from 'axios';
import type {
  PredictResponse,
  ConfigResponse,
  ExplainLogicResponse,
  DualPredictResponse,
  LareAnalysisResponse
} from '../types/detection';

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 60000,
  withCredentials: true,
});

/**
 * 获取模型配置
 */
export const fetchConfig = async (): Promise<ConfigResponse> => {
  const { data } = await apiClient.get<ConfigResponse>('/config');
  return data;
};

/**
 * 单图基础检测
 * @param file 图片文件
 */
export const predictImage = async (file: File): Promise<PredictResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await apiClient.post<PredictResponse>('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return data;
};

/**
 * 双检测器检测（基础结果 + 定位热力图兼容输出）
 * @param file 图片文件
 */
export const predictDual = async (file: File): Promise<DualPredictResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await apiClient.post<DualPredictResponse>('/predict_dual', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000
  });
  return data;
};

/**
 * LaRE 深度统计检测分析
 * @param file 图片文件
 */
export const analyzeLare = async (file: File): Promise<LareAnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const { data } = await apiClient.post<LareAnalysisResponse>('/analyze_lare', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000
  });
  return data;
};

/**
 * 逻辑解释（非流式版本）
 * @param base64Image Base64 编码的图片
 * @param isFake 是否为 AI 生成
 */
export const explainLogic = async (
  base64Image: string,
  isFake: boolean
): Promise<ExplainLogicResponse> => {
  const { data } = await apiClient.post<ExplainLogicResponse>('/explain_logic', {
    image: base64Image,
    is_fake: isFake
  }, {
    timeout: 180000
  });
  return data;
};

/**
 * 逻辑解释（流式版本）
 * @param base64Image Base64 编码的图片
 * @param isFake 是否为 AI 生成
 * @param onChunk 每接收到一个数据块时的回调函数
 */
export const explainLogicStream = async (
  base64Image: string,
  isFake: boolean,
  onChunk: (chunk: string) => void
): Promise<void> => {
  const response = await fetch('/api/explain_logic_stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: base64Image,
      is_fake: isFake
    })
  });

  if (!response.ok) {
    throw new Error(`流式请求失败: ${response.status} ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('无法读取流数据');

  const decoder = new TextDecoder('utf-8');
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      onChunk(decoder.decode(value, { stream: true }));
    }
  }
};

/**
 * 会话消息检测
 * @param messageId 消息 ID
 * @param useVl 是否使用视觉语言模型
 */
export const detectMessage = async (messageId: string, useVl = false) => {
  const res = await apiClient.post('/detect_message', {
    message_id: messageId,
    use_vl: useVl
  }, {
    timeout: 120000
  });
  return res.data;
};

// 重新导出常用类型，方便外部使用
export type { PredictResponse, ConfigResponse, DualPredictResponse, LareAnalysisResponse } from '../types/detection';
