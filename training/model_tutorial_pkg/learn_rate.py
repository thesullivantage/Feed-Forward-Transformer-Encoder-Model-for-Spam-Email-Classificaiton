import tensorflow as tf

class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    Learning rate scheduler, as described in the paper "Attention is all you need"
    '''
    def __init__(self, key_dim, warmup_steps=4000):
        super().__init__()
        self.key_dim = key_dim
        self.warmup_steps = warmup_steps
        self.d = tf.cast(self.key_dim, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'key_dim': self.key_dim,
            'warmup_steps': self.warmup_steps,
        }
        return config