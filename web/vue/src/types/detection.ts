/**
 * 检测相关类型定义
 */

/** 级联信息 */
export interface CascadeInfo {
  global_prob: number;
  local_prob: number | null;
  crop_bbox: [number, number, number, number] | null;
}

/** 单图基础检测响应 */
export interface PredictResponse {
  class_name: string;
  confidence: number;
  class_idx: number;
  probabilities: Record<string, number>;
  heatmap?: string | null;
  lare_heatmap?: string | null;
  localization_heatmap?: string | null;
  localization_score?: number | null;
  localization_area_ratio?: number | null;
  cascade_info?: CascadeInfo;
  debug?: {
    model_version: string;
    resolved_out_dir: string;
    ckpt_path?: string | null;
    cascade_enabled: boolean;
    heatmap_source?: string;
    localization_heatmap?: string | null;
    localization_enabled?: boolean;
    localization_source?: string | null;
    localization_ckpt?: string | null;
  };
}

/** 模型配置响应 */
export interface ConfigResponse {
  model_version: string;
  cascade_enabled: boolean;
  cascade_threshold: number;
  ai_confidence_threshold: number;
  lare_model_type: string;
  device: string;
  dual_detector_enabled: boolean;
  localization_enabled?: boolean;
  localization_model_name?: string;
  localization_ckpt?: string | null;
  localization_threshold?: number;
  resolved_out_dir: string;
  ckpt_path?: string | null;
}

/** 双检测器响应 */
export interface DualPredictResponse {
  prediction: string;
  confidence: number;
  detection_type: string;
  prob_global: number;
  prob_local: number | null;
  explanation: string;
  heatmap: string | null;
  localization_heatmap?: string | null;
}

/** LaRE 深度分析统计信息 */
export interface LareStatistics {
  mean?: number;
  std?: number;
  skewness?: number;
  kurtosis?: number;
  histogram_features?: number[];
  fft_features?: number[];
  noise_variance?: number;
  saturation_stats?: {
    mean: number;
    std: number;
  };
  texture_features?: Record<string, number>;
}

/** LaRE 深度分析响应 */
export interface LareAnalysisResponse {
  is_local_tampered: boolean;
  anomaly_score: number;
  statistics: LareStatistics;
  explanation: string;
  heatmap?: string | null;
}

/** 逻辑解释响应 */
export interface ExplainLogicResponse {
  report: string;
}
