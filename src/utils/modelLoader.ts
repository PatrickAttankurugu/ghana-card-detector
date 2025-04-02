// src/utils/modelLoader.ts
import * as tf from '@tensorflow/tfjs';

export class ModelLoader {
  private static instance: ModelLoader;
  private modelCache: Map<string, tf.GraphModel> = new Map();
  private loadingPromises: Map<string, Promise<tf.GraphModel>> = new Map();

  private constructor() {}

  static getInstance(): ModelLoader {
    if (!ModelLoader.instance) {
      ModelLoader.instance = new ModelLoader();
    }
    return ModelLoader.instance;
  }

  async loadModel(modelUrl: string): Promise<tf.GraphModel> {
    try {
      // Check cache first
      if (this.modelCache.has(modelUrl)) {
        return this.modelCache.get(modelUrl)!;
      }

      // Check if model is already being loaded
      if (this.loadingPromises.has(modelUrl)) {
        return await this.loadingPromises.get(modelUrl)!;
      }

      // Start new load
      const loadingPromise = tf.loadGraphModel(modelUrl)
        .then(model => {
          this.modelCache.set(modelUrl, model);
          this.loadingPromises.delete(modelUrl);
          return model;
        })
        .catch(err => {
          this.loadingPromises.delete(modelUrl);
          throw err;
        });

      this.loadingPromises.set(modelUrl, loadingPromise);
      return await loadingPromise;
    } catch (err) {
      const error = err as Error;
      throw new Error('Model loading failed: ' + error.message);
    }
  }

  async clearCache(): Promise<void> {
    try {
      // Dispose all models
      for (const model of this.modelCache.values()) {
        model.dispose();
      }
      this.modelCache.clear();
      this.loadingPromises.clear();
    } catch (err) {
      const error = err as Error;
      throw new Error('Cache clearing failed: ' + error.message);
    }
  }

  isModelCached(modelUrl: string): boolean {
    return this.modelCache.has(modelUrl);
  }

  getLoadedModelUrls(): string[] {
    return Array.from(this.modelCache.keys());
  }

  async warmupModel(modelUrl: string): Promise<void> {
    try {
      const model = await this.loadModel(modelUrl);
      const dummyTensor = tf.zeros([1, 640, 640, 3]);
      await model.predict(dummyTensor);
      tf.dispose(dummyTensor);
    } catch (err) {
      const error = err as Error;
      throw new Error('Model warmup failed: ' + error.message);
    }
  }
}