/**
 * Audio2Face-3D SDK Examples
 * 
 * This file shows examples of using the SDK in both browser and Node.js environments
 */

// ==================== BROWSER EXAMPLE ====================

// Example 1: Basic browser usage
async function browserExample() {
    // Import the SDK
    const { Audio2FaceSDK } = await import('./audio2face-sdk-browser.js');
    
    const sdk = new Audio2FaceSDK();
    
    // Load model from file
    const response = await fetch('network_actual.onnx');
    const modelBlob = await response.blob();
    await sdk.loadModel(modelBlob, { useGPU: true });
    
    // Process audio from microphone
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const source = audioContext.createMediaStreamSource(stream);
    
    // Create processor
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = async (e) => {
        const audioData = e.inputBuffer.getChannelData(0);
        const result = await sdk.processAudioChunk(audioData);
        
        // Use the blendshapes
        console.log('Blendshapes:', result.blendshapes);
        console.log('Jaw:', result.jaw);
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
}

// Example 2: Process audio file in browser
async function browserFileExample() {
    const { Audio2FaceSDK } = await import('./audio2face-sdk-browser.js');
    const sdk = new Audio2FaceSDK();
    
    // Load model
    const response = await fetch('network_actual.onnx');
    const modelBlob = await response.blob();
    await sdk.loadModel(modelBlob);
    
    // Load audio file
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    
    // Process
    const result = await sdk.processAudioFile(file);
    console.log(`Processed ${result.frameCount} frames`);
    console.log('Average jaw:', result.jaw);
}

// ==================== NODE.JS EXAMPLE ====================

// Example 3: Server-side processing
async function nodeExample() {
    const { Audio2FaceSDK } = await import('./audio2face-sdk-node.mjs');
    const fs = await import('fs');
    
    const sdk = new Audio2FaceSDK();
    
    // Load model from file path
    await sdk.loadModel('./network_actual.onnx', { useGPU: false });
    
    // Read audio file
    const audioBuffer = fs.readFileSync('input.raw');
    const audioData = new Float32Array(
        audioBuffer.buffer, 
        audioBuffer.byteOffset, 
        audioBuffer.byteLength / 4
    );
    
    // Process
    const result = await sdk.processAudioFile(audioData);
    
    // Save results
    fs.writeFileSync('output.json', JSON.stringify(result, null, 2));
    
    console.log('Processing complete!');
    console.log(`Frames: ${result.frameCount}`);
    console.log(`Jaw average: ${result.jaw}`);
    
    sdk.dispose();
}

// Example 4: Realtime processing in Node.js
async function nodeRealtimeExample() {
    const { Audio2FaceSDK } = await import('./audio2face-sdk-node.mjs');
    
    const sdk = new Audio2FaceSDK();
    await sdk.loadModel('./network_actual.onnx');
    
    // Simulate audio chunks coming in
    const chunkSize = 4096;
    let audioBuffer = new Float32Array(0);
    
    // Simulated audio stream
    async function onAudioChunk(chunk) {
        // Accumulate audio
        const newBuffer = new Float32Array(audioBuffer.length + chunk.length);
        newBuffer.set(audioBuffer);
        newBuffer.set(chunk, audioBuffer.length);
        audioBuffer = newBuffer;
        
        // Process when we have enough data
        while (audioBuffer.length >= 8320) {
            const result = await sdk.processAudioChunk(audioBuffer.slice(0, 8320));
            audioBuffer = audioBuffer.slice(4160); // Slide window
            
            // Send to client or use locally
            console.log('Got blendshapes:', result.blendshapes.slice(0, 5));
        }
    }
}

// Example 5: Using the REST API
async function apiExample() {
    // Process audio via server API
    const audioData = new Float32Array(8320);
    // ... fill audioData ...
    
    const response = await fetch('http://localhost:8765/api/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/octet-stream'
        },
        body: audioData.buffer
    });
    
    const result = await response.json();
    console.log('Blendshapes:', result.blendshapes);
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { browserExample, nodeExample };
}