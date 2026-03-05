/**
 * AudioWorklet processor for mic capture.
 * Runs on the audio thread for consistent timing.
 * Captures audio at native sample rate and outputs PCM int16 chunks.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        this.chunkSize = 320; // 20ms at 16kHz
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const samples = input[0]; // Float32Array

        // Convert to int16 and buffer
        for (let i = 0; i < samples.length; i++) {
            const s = Math.max(-1, Math.min(1, samples[i]));
            this.buffer.push(s * 32767 | 0);
        }

        // Send chunks
        while (this.buffer.length >= this.chunkSize) {
            const chunk = this.buffer.splice(0, this.chunkSize);
            const pcm = new Int16Array(chunk);
            this.port.postMessage({ type: 'audio', pcm: pcm.buffer }, [pcm.buffer]);
        }

        return true;
    }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
