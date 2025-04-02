// src/utils/imageProcessing.ts
import * as tf from '@tensorflow/tfjs';
import { PreprocessedImage } from '../types';

export class ImageProcessor {
  private readonly TARGET_SIZE = 640;

  async preprocessImage(
    image: HTMLImageElement | HTMLVideoElement
  ): Promise<PreprocessedImage> {
    return tf.tidy(() => {
      // Convert image to tensor
      const imageTensor = tf.browser.fromPixels(image);
      
      // Calculate scaling factors
      const scale = {
        x: image.width / this.TARGET_SIZE,
        y: image.height / this.TARGET_SIZE
      };

      // Resize and normalize
      const resized = tf.image
        .resizeBilinear(imageTensor, [this.TARGET_SIZE, this.TARGET_SIZE])
        .div(255.0)
        .expandDims(0);

      return {
        tensor: resized,
        scale
      };
    });
  }

  postprocessDetections(
    predictions: tf.Tensor,
    scale: { x: number; y: number },
    scoreThreshold: number,
    maxDetections: number
  ) {
    const boxes = predictions.arraySync()[0];
    const detections = [];

    for (let i = 0; i < boxes.length; i++) {
      const [x1, y1, x2, y2, confidence] = boxes[i];

      if (confidence > scoreThreshold) {
        detections.push({
          box: {
            x1: x1 * scale.x,
            y1: y1 * scale.y,
            x2: x2 * scale.x,
            y2: y2 * scale.y
          },
          confidence,
          class: 'ghana_card'
        });
      }
    }

    // Sort by confidence and limit detections
    return detections
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, maxDetections);
  }
}