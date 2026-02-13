# Audio2Face-3D Browser & Node.js SDK

A buildless, universal SDK for converting audio to 3D facial animation using NVIDIA's Audio2Face-3D models. Works in both browsers and Node.js environments.

## Features

- **Universal**: Works in browsers (with WebGPU/WebAssembly) and Node.js (with native CPU/GPU acceleration)
- **Buildless Browser**: Pure JavaScript/ES modules, no build step required for browser usage
- **Realtime Processing**: Stream audio from microphone and get live blendshape output
- **Non-realtime Processing**: Process audio files in batch mode
- **Server API**: Express server with REST endpoints for server-side processing
- **Zero CDN Dependencies**: All files served locally

## Quick Start

### 1. Install Dependencies (for Node.js/Server)

```bash
npm install
```

### 2. Download the Model

Download `network.onnx` from [Hugging Face](https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire) and place it in the project root as `network_actual.onnx`.

### 3. Start the Server

```bash
npm start
# or
node server.js
```

The server will:
- Serve the browser UI at `http://localhost:8765`
- Provide REST API endpoints for server-side processing
- Load the model automatically

## Usage

### Browser (Client-Side)

Open `http://localhost:8765` in your browser. The model loads automatically and you can:
- Use your microphone for real-time processing
- Upload audio files for batch processing
- View live blendshape outputs

**HTML:**
```html
<!DOCTYPE html>
<html>
<head>
    <script src="onnxruntime-web.min.js"></script>
    <script type="module">
        import { Audio2FaceSDK } from './audio2face-sdk-browser.js';
        
        const sdk = new Audio2FaceSDK();
        
        // Load model
        const response = await fetch('network_actual.onnx');
        const modelBlob = await response.blob();
        await sdk.loadModel(modelBlob);
        
        // Process audio
        const result = await sdk.processAudioChunk(audioData);
        console.log(result.blendshapes);
    </script>
</head>
</html>
```

### Node.js (Server-Side)

```javascript
import { Audio2FaceSDK } from './audio2face-sdk-node.mjs';
import fs from 'fs';

const sdk = new Audio2FaceSDK();

// Load model
await sdk.loadModel('./network_actual.onnx');

// Process audio file
const audioBuffer = fs.readFileSync('audio.raw');
const audioData = new Float32Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.length / 4);

const result = await sdk.processAudioFile(audioData);
console.log(result.blendshapes);
```

### REST API

The server provides the following endpoints:

#### Health Check
```bash
GET /api/health
```

Response:
```json
{
  "status": "ok",
  "modelLoaded": true,
  "timestamp": 1234567890
}
```

#### Process Audio (Batch)
```bash
POST /api/process
Content-Type: application/octet-stream

<raw audio data>
```

Response:
```json
{
  "blendshapes": [...],
  "jaw": 0.5,
  "eyes": {...},
  "frameCount": 30
}
```

#### Process Audio Chunk (Realtime)
```bash
POST /api/process-chunk
Content-Type: application/octet-stream

<audio chunk>
```

Response:
```json
{
  "blendshapes": [...],
  "jaw": 0.5,
  "eyes": {...},
  "timestamp": 1234567890
}
```

## API Reference

### `Audio2FaceSDK`

#### Constructor
```javascript
const sdk = new Audio2FaceSDK();
```

#### Methods

##### `loadModel(modelFile, options)`
Load an ONNX model file.
- `modelFile`: File (browser), string path (Node.js), ArrayBuffer, Blob, or Buffer
- `options.useGPU`: Enable GPU acceleration (default: true in browser, false in Node.js)

##### `processAudioFile(audioData)`
Process an entire audio file and return averaged blendshape values.
- `audioData`: Float32Array (browser) or Buffer/Float32Array (Node.js)
- Returns: `{ blendshapes, jaw, eyes, frameCount }`

##### `processAudioChunk(audioData)`
Process a single audio chunk for realtime streaming.
- `audioData`: Float32Array
- Returns: `{ blendshapes, jaw, eyes, timestamp }`

##### `setSmoothing(factor)`
Set temporal smoothing factor (0-1, default: 0.3).

##### `dispose()`
Release model resources.

## Output Format

```javascript
{
  blendshapes: [
    { name: "blendshape_0", value: 0.5 },
    { name: "blendshape_1", value: 0.2 },
    // ... 52 blendshapes total
  ],
  jaw: 0.3,           // Jaw openness
  eyes: {             // Eye gaze (if available)
    leftX: 0.1,
    leftY: 0.2,
    rightX: 0.1,
    rightY: 0.2
  },
  frameCount: 30,     // Number of frames processed (batch mode)
  timestamp: 12345    // Timestamp (realtime mode)
}
```

## Audio Requirements

- Sample rate: 16kHz (automatically resampled if different)
- Format: Mono (single channel), Float32
- Chunk size for realtime: 8320 samples (~520ms at 16kHz)
- Hop size: 4160 samples (~260ms at 16kHz)

## Browser Compatibility

- Chrome/Edge 113+ (WebGPU support)
- Firefox (WebAssembly fallback)
- Safari (WebAssembly fallback)

## File Structure

```
├── audio2face-sdk-browser.js    # Browser SDK (uses onnxruntime-web)
├── audio2face-sdk-node.mjs      # Node.js SDK (uses onnxruntime-node)
├── index.html                    # Browser demo UI
├── test.html                     # Browser test page
├── test-node.mjs                 # Node.js test script
├── server.js                     # Express server with API
├── server.py                     # Simple Python static server
├── package.json                  # NPM configuration
├── network_actual.onnx           # The model file (download separately)
├── onnxruntime-web.min.js        # ONNX Runtime for browsers
└── ort-wasm-simd-threaded.wasm   # WebAssembly binary
```

## Testing

### Browser Test
```bash
python3 server.py
# Open http://localhost:8765/test.html
```

### Node.js Test
```bash
npm test
# or
node test-node.mjs
```

## Troubleshooting

### "Model not loaded" error
Ensure `network_actual.onnx` exists in the project directory. Download it from [Hugging Face](https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire).

### CORS errors
Use the provided `server.py` or `server.js` which include proper CORS headers and MIME type configuration.

### WebAssembly MIME type errors
The server must serve `.wasm` files with `application/wasm` MIME type. Use the provided servers.

### GPU not available
The SDK automatically falls back to CPU if WebGPU or CUDA is not available.

## License

This SDK code is provided as-is. The Audio2Face-3D models are licensed under the NVIDIA Open Model License.

## Credits

- [NVIDIA Audio2Face-3D](https://github.com/NVIDIA/Audio2Face-3D)
- [ONNX Runtime](https://onnxruntime.ai/)