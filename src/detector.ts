import * as tf from '@tensorflow/tfjs';
import { DetectionResult, GhanaCardDetectorConfig } from './types';
import { ImageProcessor } from './utils/imageProcessing';
import { ModelLoader } from './utils/modelLoader';

export class GhanaCardDetector {
  private model: tf.GraphModel | null = null;
  private modelUrl: string;
  private version: string;
  private scoreThreshold: number;
  private maxDetections: number;
  private imageProcessor: ImageProcessor;
  private modelLoader: ModelLoader;

  constructor(config?: GhanaCardDetectorConfig) {
    this.modelUrl = config?.modelUrl || 
      'https://github.com/PatrickAttankurugu/ghana-card-detector/releases/download/v1.0.0/model.tflite';
    this.version = config?.version || '1.0.0';
    this.scoreThreshold = config?.scoreThreshold || 0.5;
    this.maxDetections = config?.maxDetections || 5;
    this.imageProcessor = new ImageProcessor();
    this.modelLoader = ModelLoader.getInstance();
  }

  async initialize(): Promise<void> {
    try {
      this.model = await this.modelLoader.loadModel(this.modelUrl);
      console.log(`Ghana Card Detector v${this.version} initialized successfully`);
    } catch (error) {
      throw new Error(`Model initialization failed: ${error.message}`);
    }
  }

  async detect(
    image: HTMLImageElement | HTMLVideoElement
  ): Promise<DetectionResult[]> {
    if (!this.model) {
      throw new Error('Model not initialized. Call initialize() first.');
    }

    // Preprocess image
    const preprocessed = await this.imageProcessor.preprocessImage(image);

    // Run inference
    const predictions = await this.model.predict(preprocessed.tensor) as tf.Tensor;

    // Post-process results
    const detections = this.imageProcessor.postprocessDetections(
      predictions,
      preprocessed.scale,
      this.scoreThreshold,
      this.maxDetections
    );

    // Cleanup
    tf.dispose([preprocessed.tensor, predictions]);

    return detections;
  }

  getVersion(): string {
    return this.version;
  }
}