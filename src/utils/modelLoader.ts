// src/utils/modelLoader.ts
import * as tf from '@tensorflow/tfjs';

export class ModelLoader {
  private static instance: ModelLoader;
  private modelCache: Map<string, tf.GraphModel> = new Map();

  static getInstance(): ModelLoader {
    if (!ModelLoader.instance) {
      ModelLoader.instance = new ModelLoader();
    }
    return ModelLoader.instance;
  }

  async loadModel(modelUrl: string): Promise<tf.GraphModel> {
    if (this.modelCache.has(modelUrl)) {
      return this.modelCache.get(modelUrl)!;
    }

    try {
      const model = await tf.loadGraphModel(modelUrl);
      this.modelCache.set(modelUrl, model);
      return model;
    } catch (error) {
      throw new Error(`Failed to load model: ${error.message}`);
    }
  }

  clearCache() {
    this.modelCache.clear();
  }
}