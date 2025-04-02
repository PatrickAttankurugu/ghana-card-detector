// src/utils/imageProcessing.ts
import * as tf from '@tensorflow/tfjs';
import { PreprocessedImage, DetectionResult } from '../types';

export class ImageProcessor {
  private readonly TARGET_SIZE = 640;

  async preprocessImage(
    image: HTMLImageElement | HTMLVideoElement | ImageData
  ): Promise<PreprocessedImage> {
    try {
      return tf.tidy(() => {
        // Convert image to tensor
        let imageTensor = tf.browser.fromPixels(image);
        
        // Get original dimensions
        const originalHeight = imageTensor.shape[0];
        const originalWidth = imageTensor.shape[1];

        // Calculate scaling factors
        const scale = {
          x: originalWidth / this.TARGET_SIZE,
          y: originalHeight / this.TARGET_SIZE
        };

        // Resize and normalize
        const resized = tf.image
          .resizeBilinear(imageTensor, [this.TARGET_SIZE, this.TARGET_SIZE])
          .div(255.0)
          .expandDims(0);

        return {
          tensor: resized,
          scale,
          originalSize: {
            width: originalWidth,
            height: originalHeight
          }
        };
      });
    } catch (err) {
      const error = err as Error;
      throw new Error('Image preprocessing failed: ' + error.message);
    }
  }

  postprocessDetections(
    predictions: tf.Tensor,
    scale: { x: number; y: number },
    scoreThreshold: number,
    maxDetections: number
  ): DetectionResult[] {
    try {
      const boxes = predictions.arraySync() as number[][][];
      const detections: DetectionResult[] = [];

      for (let i = 0; i < boxes[0].length; i++) {
        const [x1, y1, x2, y2, confidence] = boxes[0][i];

        if (confidence > scoreThreshold) {
          detections.push({
            box: {
              x1: Math.round(x1 * scale.x),
              y1: Math.round(y1 * scale.y),
              x2: Math.round(x2 * scale.x),
              y2: Math.round(y2 * scale.y)
            },
            confidence: parseFloat(confidence.toFixed(4)),
            class: 'ghana_card'
          });
        }
      }

      // Sort by confidence and limit detections
      return detections
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, maxDetections);
    } catch (err) {
      const error = err as Error;
      throw new Error('Detection postprocessing failed: ' + error.message);
    }
  }

  drawDetections(
    canvas: HTMLCanvasElement,
    detections: DetectionResult[]
  ): void {
    try {
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Unable to get canvas context');

      // Clear previous drawings
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      detections.forEach(detection => {
        const { box, confidence } = detection;
        
        // Draw bounding box
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(
          box.x1,
          box.y1,
          box.x2 - box.x1,
          box.y2 - box.y1
        );

        // Draw label
        ctx.fillStyle = '#00FF00';
        ctx.font = '16px Arial';
        ctx.fillText(
          `Ghana Card: ${(confidence * 100).toFixed(1)}%`,
          box.x1,
          box.y1 - 5
        );
      });
    } catch (err) {
      const error = err as Error;
      throw new Error('Drawing detections failed: ' + error.message);
    }
  }

  // Utility method to validate image input
  private validateImageInput(
    image: HTMLImageElement | HTMLVideoElement | ImageData
  ): void {
    if (image instanceof HTMLImageElement && !image.complete) {
      throw new Error('Image not loaded');
    }

    if (
      image instanceof HTMLVideoElement && 
      (image.readyState < 2 || !image.videoWidth || !image.videoHeight)
    ) {
      throw new Error('Video not ready');
    }
  }

  // Method to cleanup tensors
  dispose(tensors: tf.Tensor[]): void {
    tensors.forEach(tensor => {
      if (tensor && tensor.dispose) {
        tensor.dispose();
      }
    });
  }
}