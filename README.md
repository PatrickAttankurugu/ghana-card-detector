# Ghana Card Detector

A JavaScript/TypeScript package for detecting Ghana cards in images and video streams using TensorFlow.js and a custom-trained YOLOv8 model.

## Features

- Real-time Ghana card detection
- Support for both image and video input
- High accuracy (97.48% precision, 100% recall)
- TypeScript support
- Built-in visualization tools
- Memory-efficient model loading
- Customizable detection parameters

## Installation

```bash
npm install ghana-card-detector
Basic Usage

Typescript

import { GhanaCardDetector } from 'ghana-card-detector';

// Initialize detector
const detector = new GhanaCardDetector();
await detector.initialize();

// Detect from image
const img = document.querySelector('img');
const results = await detector.detect(img);

// Process results
results.forEach(detection => {
  console.log('Detection:', {
    confidence: detection.confidence,
    boundingBox: detection.box
  });
});
Example with Video Stream
typescriptCopyimport { GhanaCardDetector } from 'ghana-card-detector';

async function setupDetector() {
  // Initialize detector
  const detector = new GhanaCardDetector();
  await detector.initialize();

  // Access camera
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  const video = document.querySelector('video');
  video.srcObject = stream;

  // Create canvas for visualization
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);

  // Detection loop
  async function detectFrame() {
    const detections = await detector.detect(video);
    
    // Draw results
    detector.drawDetections(canvas, detections);
    
    requestAnimationFrame(detectFrame);
  }

  detectFrame();
}
Advanced Configuration
typescriptCopyconst detector = new GhanaCardDetector({
  modelUrl: 'custom-model-url',  // Custom model URL
  version: '1.0.0',             // Model version
  scoreThreshold: 0.5,          // Detection confidence threshold
  maxDetections: 5              // Maximum number of detections
});
API Reference
GhanaCardDetector
Constructor
typescriptCopyconstructor(config?: {
  modelUrl?: string;
  version?: string;
  scoreThreshold?: number;
  maxDetections?: number;
})
Methods
initialize(): Promise<void>
Loads the model and prepares it for detection.
detect(image: HTMLImageElement | HTMLVideoElement): Promise<DetectionResult[]>
Performs detection on the input image/video frame.
drawDetections(canvas: HTMLCanvasElement, detections: DetectionResult[]): void
Visualizes detections on a canvas element.
Types
typescriptCopyinterface DetectionResult {
  box: {
    x1: number;  // Top-left x coordinate
    y1: number;  // Top-left y coordinate
    x2: number;  // Bottom-right x coordinate
    y2: number;  // Bottom-right y coordinate
  };
  confidence: number;  // Detection confidence (0-1)
  class: string;      // Always "ghana_card"
}
Integration with Popular Frameworks
Next.js Example
typescriptCopyimport { useEffect, useRef, useState } from 'react';
import { GhanaCardDetector } from 'ghana-card-detector';

export default function GhanaCardScanner() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [detector, setDetector] = useState<GhanaCardDetector | null>(null);

  useEffect(() => {
    const initDetector = async () => {
      const newDetector = new GhanaCardDetector();
      await newDetector.initialize();
      setDetector(newDetector);
    };
    initDetector();
  }, []);

  const startScanning = async () => {
    if (!videoRef.current || !canvasRef.current || !detector) return;

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoRef.current.srcObject = stream;

    const detectFrame = async () => {
      const detections = await detector.detect(videoRef.current!);
      detector.drawDetections(canvasRef.current!, detections);
      requestAnimationFrame(detectFrame);
    };

    detectFrame();
  };

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline />
      <canvas ref={canvasRef} />
      <button onClick={startScanning}>Start Scanning</button>
    </div>
  );
}
Performance Metrics
The model achieves the following performance metrics:

Precision: 97.48%
Recall: 100%
mAP50: 99.50%
mAP50-95: 97.64%

Browser Compatibility

Chrome (recommended)
Firefox
Safari
Edge

Requirements

Node.js >= 14
Modern browser with WebGL support
Webcam (for video detection)