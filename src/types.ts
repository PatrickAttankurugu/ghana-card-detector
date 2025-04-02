// src/types.ts
export interface Box {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  }
  
  export interface DetectionResult {
    box: Box;
    confidence: number;
    class: string;
  }
  
  export interface GhanaCardDetectorConfig {
    modelUrl?: string;
    version?: string;
    scoreThreshold?: number;
    maxDetections?: number;
  }
  
  export interface PreprocessedImage {
    tensor: any;
    scale: {
      x: number;
      y: number;
    };
  }