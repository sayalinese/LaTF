export interface PredictResponse {
  class_name: string;
  confidence: number;
  class_idx: number;
  probabilities: {
    [key: string]: number;
  };
  heatmap?: string | null;
  cascade_info?: {
    global_prob: number;
    local_prob: number | null;
    crop_bbox: number[] | null;
  };
  debug?: {
    model_version: string;
    resolved_out_dir: string;
    ckpt_path: string | null;
    cascade_enabled: boolean;
    cascade_threshold: number;
    ai_confidence_threshold: number;
    heatmap_source?: string;
    heatmap_map_shape?: number[];
  };
}

export interface ConfigResponse {
  model_version: string;
  cascade_enabled: boolean;
  resolved_out_dir: string;
  ckpt_path: string | null;
}
