# Training Notes - Alterations from ML Mastery Tutorial

## For NaN loss:
- Still using trainable position embedding. Basic working theory is that using the sinusoidal positon encoding does not provide the same amount of information that the trainable encoding does
    - That is, the dense matrix mapped to with layers. Embedding is trained in the gradient descent process, while the sinusoidal one is not. 

## Tuning steps that have produced good results:
- Gradient Clipping:
    - Floor (limit) gradients via Adam optimizer
    - Observed wildly varying gradients (also NaN loss problem)
        - In the "hills and valleys" of our loss surface: the update steps can 

- Exponential Decay Learning Rate Schedule
    - Based on observations of good training results: 
        - Apply a relatively high initial learning rate for burn-in/warmup 
            - Do not use linearly increasing LR schedule for warmup steps: can move onto this type of tuning in the future
        - Decrease (exponentially) to half of initial learning rate after all training steps (manually set)
        