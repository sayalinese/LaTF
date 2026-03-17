export interface PredictResponse {
  class_name: string;
  confidence: number;
  class_idx: number;
  probabilities: Record<string, number>;
  heatmap?: string | null;
  debug?: Record<string, any>;
}

export interface ConfigResponse {
  model_version: string;
  cascade_enabled: boolean;
  cascade_threshold: number;
  ai_confidence_threshold: number;
  lare_model_type: string;
  device: string;
  dual_detector_enabled: boolean;
  resolved_out_dir: string;
  ckpt_path?: string | null;
}

export interface ExplainLogicResponse {
  report: string;
}