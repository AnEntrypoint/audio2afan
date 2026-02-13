import express from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { Audio2FaceSDK } from './audio2face-sdk-node.mjs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8765;

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.raw({ type: 'audio/*', limit: '50mb' }));

// Initialize SDK
const sdk = new Audio2FaceSDK();
let modelLoaded = false;

// Load model on startup
async function loadModel() {
    try {
        const modelPath = path.join(__dirname, 'network_actual.onnx');
        if (!fs.existsSync(modelPath)) {
            console.error('Model file not found:', modelPath);
            console.log('Please ensure network_actual.onnx exists in the project directory');
            return false;
        }
        
        console.log('Loading model...');
        await sdk.loadModel(modelPath, { useGPU: false });
        modelLoaded = true;
        console.log('✓ Model loaded successfully');
        console.log(`  - Input names: ${sdk.session.inputNames.join(', ')}`);
        console.log(`  - Output names: ${sdk.session.outputNames.join(', ')}`);
        return true;
    } catch (err) {
        console.error('Failed to load model:', err.message);
        return false;
    }
}

// API Routes

// Health check
app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        modelLoaded,
        timestamp: Date.now()
    });
});

// Process audio (server-side)
app.post('/api/process', async (req, res) => {
    if (!modelLoaded) {
        return res.status(503).json({ error: 'Model not loaded' });
    }
    
    try {
        // Convert buffer to Float32Array
        const audioBuffer = req.body;
        const audioData = new Float32Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength / 4);
        
        const result = await sdk.processAudioFile(audioData);
        
        res.json(result);
    } catch (err) {
        console.error('Processing error:', err);
        res.status(500).json({ error: err.message });
    }
});

// Process audio chunk (realtime)
app.post('/api/process-chunk', async (req, res) => {
    if (!modelLoaded) {
        return res.status(503).json({ error: 'Model not loaded' });
    }
    
    try {
        const audioBuffer = req.body;
        const audioData = new Float32Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength / 4);
        
        const result = await sdk.processAudioChunk(audioData);
        
        res.json(result);
    } catch (err) {
        console.error('Processing error:', err);
        res.status(500).json({ error: err.message });
    }
});

// Static files with custom MIME types
app.use(express.static(__dirname, {
    setHeaders: (res, path) => {
        if (path.endsWith('.mjs')) {
            res.setHeader('Content-Type', 'application/javascript');
        } else if (path.endsWith('.wasm')) {
            res.setHeader('Content-Type', 'application/wasm');
        }
    }
}));

// Serve index.html for root
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
async function start() {
    await loadModel();
    
    app.listen(PORT, () => {
        console.log(`\nServer running on http://localhost:${PORT}`);
        console.log('\nAvailable endpoints:');
        console.log(`  - Browser UI: http://localhost:${PORT}`);
        console.log(`  - Health check: http://localhost:${PORT}/api/health`);
        console.log(`  - Process audio: POST http://localhost:${PORT}/api/process`);
        console.log(`  - Process chunk: POST http://localhost:${PORT}/api/process-chunk`);
        console.log('\nPress Ctrl+C to stop\n');
    });
}

start().catch(err => {
    console.error('Server failed to start:', err);
    process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\nShutting down...');
    sdk.dispose();
    process.exit(0);
});