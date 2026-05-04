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

export interface ExplainLogicResponse {
  report: string;
}