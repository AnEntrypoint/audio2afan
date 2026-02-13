import { Audio2FaceSDK } from './audio2face-sdk-node.mjs';
import fs from 'fs';
import path from 'path';

async function runTest() {
    console.log('Audio2Face-3D Node.js SDK Test');
    console.log('================================\n');
    
    try {
        const sdk = new Audio2FaceSDK();
        
        // Check if model exists
        const modelPath = './network_actual.onnx';
        if (!fs.existsSync(modelPath)) {
            console.error(`Error: Model file not found at ${modelPath}`);
            console.log('Please ensure network_actual.onnx exists in the current directory.');
            process.exit(1);
        }
        
        console.log('Loading model...');
        const startTime = Date.now();
        await sdk.loadModel(modelPath, { useGPU: false });
        const loadTime = Date.now() - startTime;
        console.log(`Model loaded in ${loadTime}ms`);
        console.log(`Input names: ${sdk.session.inputNames.join(', ')}`);
        console.log(`Output names: ${sdk.session.outputNames.join(', ')}\n`);
        
        // Test with synthetic audio
        console.log('Running inference test with synthetic audio...');
        const testAudio = new Float32Array(8320);
        for (let i = 0; i < 8320; i++) {
            testAudio[i] = Math.sin(i * 0.1) * 0.5;
        }
        
        const result = await sdk.processAudioChunk(testAudio);
        
        console.log('\nTest Results:');
        console.log(`- Blendshapes detected: ${result.blendshapes.length}`);
        console.log(`- First 5 blendshape values:`);
        result.blendshapes.slice(0, 5).forEach((bs, i) => {
            console.log(`  ${bs.name}: ${bs.value.toFixed(4)}`);
        });
        console.log(`- Jaw value: ${result.jaw.toFixed(4)}`);
        console.log(`- Timestamp: ${result.timestamp}`);
        
        // Test batch processing
        console.log('\n\nTesting batch processing...');
        const batchAudio = new Float32Array(8320 * 3); // 3 frames worth
        for (let i = 0; i < batchAudio.length; i++) {
            batchAudio[i] = Math.sin(i * 0.05) * 0.5;
        }
        
        const batchResult = await sdk.processAudioFile(batchAudio);
        console.log(`Batch processing complete: ${batchResult.frameCount} frames`);
        console.log(`Average jaw value: ${batchResult.jaw.toFixed(4)}`);
        
        console.log('\n✓ All tests passed successfully!');
        
        sdk.dispose();
    } catch (err) {
        console.error('\n✗ Test failed:', err.message);
        console.error(err);
        process.exit(1);
    }
}

runTest();