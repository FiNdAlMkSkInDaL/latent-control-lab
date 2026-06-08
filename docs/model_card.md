# Model Card Notes

## Default checkpoint

`google/gemma-2-2b-it` is the intended Colab-tier checkpoint. Use `sshleifer/tiny-gpt2` only for smoke tests.

## Runtime assumptions

- Frozen model parameters.
- No generation in action routing.
- Fixed prompt template across feature extraction and inference.
- Final-token feature vector from pre-lm_head activation.

## Hardware notes

- CPU: use tiny models for tests only.
- Free Colab GPU: use 2B-class models and small batches.
- 8B-class models: use quantization and lower batch sizes.
