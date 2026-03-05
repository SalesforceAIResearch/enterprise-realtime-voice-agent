/**
 * AudioWorklet processor for streaming playback.
 * Receives PCM chunks from main thread and plays them smoothly.
 */
class AudioPlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.queue = [];
        this.currentBuffer = null;
        this.currentOffset = 0;

        this.port.onmessage = (e) => {
            if (e.data.type === 'audio') {
                const int16 = new Int16Array(e.data.pcm);
                const float32 = new Float32Array(int16.length);
                for (let i = 0; i < int16.length; i++) {
                    float32[i] = int16[i] / 32768;
                }
                this.queue.push(float32);
            } else if (e.data.type === 'clear') {
                this.queue = [];
                this.currentBuffer = null;
                this.currentOffset = 0;
            }
        };
    }

    process(inputs, outputs) {
        const output = outputs[0][0];
        let written = 0;

        while (written < output.length) {
            if (!this.currentBuffer || this.currentOffset >= this.currentBuffer.length) {
                if (this.queue.length > 0) {
                    this.currentBuffer = this.queue.shift();
                    this.currentOffset = 0;
                } else {
                    // Fill rest with silence
                    for (let i = written; i < output.length; i++) output[i] = 0;
                    break;
                }
            }

            const remaining = this.currentBuffer.length - this.currentOffset;
            const needed = output.length - written;
            const toCopy = Math.min(remaining, needed);

            for (let i = 0; i < toCopy; i++) {
                output[written + i] = this.currentBuffer[this.currentOffset + i];
            }

            this.currentOffset += toCopy;
            written += toCopy;
        }

        if (this.queue.length === 0 && (!this.currentBuffer || this.currentOffset >= this.currentBuffer.length)) {
            this.port.postMessage({ type: 'playback_ended' });
        }

        return true;
    }
}

registerProcessor('audio-playback-processor', AudioPlaybackProcessor);
