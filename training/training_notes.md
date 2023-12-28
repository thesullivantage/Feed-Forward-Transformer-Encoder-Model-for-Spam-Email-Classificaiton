# Training Notes

## On observing exploding gradients (NaN loss) with sinusoidal positional text embedding:
- Still using trainable position embedding.
    - That is, the dense matrix mapped to with layers.
    - Embedding is trained in the gradient descent process, while the sinusoidal embeddings are fixed by position.
    - Potential for more information (an extra layer of training) with more spatial/time complexity.

## Tuning steps that have produced good results:
- Gradient Clipping:
    - Floor (limit) gradients via Adam optimizer in each layer.

- Exponential Decay Learning Rate Schedule
    - Based on observations of good training results: 
        - Apply a relatively high initial learning rate for burn-in/warmup 
            - Do not use linearly increasing learning rate schedule for warmup steps: can move onto this type of tuning in the future
        - Decrease (exponentially) to half of the initial learning rate after all training steps (epochs), based on empirical observations of performance.
        
