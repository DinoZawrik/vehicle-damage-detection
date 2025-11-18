export interface Detection {
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  severity?: string;
}

export interface Analysis {
  severity: string;
  damage_count: number;
  damage_types: Record<string, number>;
  confidence_avg: number;
}

export interface CostEstimate {
  min: number;
  max: number;
  currency: string;
  total: number;
}

export interface ImageInfo {
  filename: string;
  width: number;
  height: number;
}

export interface ModelInfo {
  name: string;
  confidence_threshold: number;
}

export interface ApiResponse {
  success: boolean;
  timestamp: string;
  image_info: ImageInfo;
  detections: Detection[];
  analysis: Analysis;
  cost_estimate: CostEstimate;
  processing_time: number;
  model: ModelInfo;
}
