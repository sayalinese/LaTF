export interface CascadeInfo {
  global_prob: number;
  local_prob: number | null;
  crop_bbox: [number, number, number, number] | null;
}

export interface PredictResponse {
  class_name: string;
  confidence: number;
  class_idx: number;
  probabilities: Record<string, number>;
  heatmap?: string | null;
  cascade_info?: CascadeInfo;
  debug?: {
    model_version: string;
    resolved_out_dir: string;
    ckpt_path?: string | null;
    cascade_enabled: boolean;
    heatmap_source?: string;
    localization_heatmap?: string | null;
  };
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