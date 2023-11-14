import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class TransformerBase(tf.keras.layers.Layer):
    '''
    Base functions for transformer model construction.
    '''
    def __init__(self):
        super().__init__()
        return None
    
    @classmethod
    def pos_enc_matrix(L, d, n=10000):
        """Create positional encoding matrix
    
        Args:
            L: Input dimension (length)
            d: Output dimension (depth), even only
            n: Constant for the sinusoidal functions
    
        Returns:
            numpy matrix of floats of dimension L-by-d. At element (k,2i) the value
            is sin(k/n^(2i/d)) while at element (k,2i+1) the value is cos(k/n^(2i/d))
        """
        assert d % 2 == 0, "Output dimension needs to be an even integer"
        d2 = d//2
        P = np.zeros((L, d))
        k = np.arange(L).reshape(-1, 1)     # L-column vector
        i = np.arange(d2).reshape(1, -1)    # d-row vector
        denom = np.power(n, -i/d2)          # n**(-2*i/d)
        args = k * denom                    # (L,d) matrix
        P[:, ::2] = np.sin(args)
        P[:, 1::2] = np.cos(args)
        return P
    
class TokenPositionEmbeddingOriginal(TransformerBase):
    '''
        Token and position embeddings. 
        Token embedding layer: trainable lookup matrix | Positional encoding from sinusoids as in "Attention is All You Need".
        Token embedding expects vectorized representations of text passed in. Index of sequence implicit for positional encoding.
    '''
    def __init__(self,  sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # token embedding layer: create trainable lookup table (matrix) for vocab_size words (dim: (vocab_size, embed_dim) ) 
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero=True, name='embed_vocab'
        )
        # positional embedding layer: a matrix of constant sinusoid values - 2 for each position of input sequence (see transformer paper)
        matrix = self.pos_enc_matrix(self.sequence_length, self.embed_dim)
        self.position_embeddings = tf.constant(matrix, dtype="float32")

    def call(self, inputs):
        embedded_tokens = self.token_embeddings(inputs)
        # input tokens embedding vector superimposed with position encoding vector
        return embedded_tokens + self.position_embeddings
    
    def compute_mask(self, *args, **kwargs):
        # can (optionally) define a causal mask to not attend to words in sequence not yet reached (i.e. in translation tasks)
        return self.token_embeddings.compute_mask(*args, **kwargs)

class TokenPositionEmbeddingLookup(TransformerBase):
    '''
        Token and position embedding layer. 
        Token encoding and positional encoding: both trainable lookup matrices.
        Token embedding expects vectorized representations of text passed in. Index of sequence implicit for positional encoding.
    '''
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        # token embedding layer: create trainable lookup table (matrix) for vocab_size words (dim: (vocab_size, embed_dim))
        # AND the same for sequence_length many tokens (dim: (sequence_length, embed_dim))
        self.token_embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name='embed_vocab')
        self.pos_embed = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim, name='embed_sequence')

    def call(self, x):
        sequence_length = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=sequence_length, delta=1)
        positions = self.pos_embed(positions)
        token_embeddings = self.token_embed(x)
        return token_embeddings + positions
    
    # def compute_mask(self, *args, **kwargs):
    #     # can define a causal mask to not attend to words in sequence not yet reached (i.e. in translation tasks)
    #     return self.token_embed.compute_mask(*args, **kwargs)
    
class SelfAttention(TransformerBase):
    '''
        Self-attention layer.
        Self-attention is comparable, in classical statistical sense, to the covariance matrix of a data sequence.
        Here, though, the layers is also trained on how to attend to each position/pair of positions. 
    '''
    def __init__(self, input_shape, prefix="att", dropout=0.1, clipvalue=0.15, **kwargs):
        super().__init__()

        self.inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32',
                                        name=f"{prefix}_in1")
        # self-attention layer
        self.attention = tf.keras.layers.MultiHeadAttention(name=f"{prefix}_attn1", **kwargs)
        self.drop = tf.keras.layers.Dropout(rate=dropout, name=f"{prefix}_dropout")
        self.norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm1", epsilon=1e-6)
        self.add = tf.keras.layers.Add(name=f"{prefix}_add1")
        # functional API to connect input to output
        self.attout = self.attention(key=self.inputs, value=self.inputs, query=self.inputs)
        self.outputs = self.norm(self.add([self.inputs, self.attout]))
        # create model and return
        # MARK DEV: CALL SUBSTITUTE
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name=f"{prefix}_att")
    
    def __call__(self, inputs):
        return self.model(inputs)

class FeedForward(TransformerBase):
    """
        Feed-forward layers at transformer encoder and decoder, post-SelfAttention layer (instance). 
        Assumes input is from output of an attention layer with add & norm
        Output produced is the final output of a single encoder or decoder block

        Args:
            model_dim (int): Output dimension of the feed-forward layer, which
                                is also the output dimension of the encoder/decoder
                                block
            ff_dim (int): Internal dimension of the feed-forward layer
            dropout (float): Dropout rate
            prefix (str): The prefix added to the layer names
    """
    def __init__(self, input_shape, model_dim, ff_dim, dropout=0.1, clipvalue=0.15, prefix="ff"):
        super().__init__()
        # create layers
        self.inputs = tf.keras.layers.Input(shape=input_shape, dtype='float32',
                                        name=f"{prefix}_in3")
        self.dense1 = tf.keras.layers.Dense(ff_dim, name=f"{prefix}_ff1", activation="relu")
        self.dense2 = tf.keras.layers.Dense(model_dim, name=f"{prefix}_ff2")
        self.drop = tf.keras.layers.Dropout(dropout, name=f"{prefix}_drop")
        self.add = tf.keras.layers.Add(name=f"{prefix}_add3")
        self.ffout = self.drop(self.dense2(self.dense1(self.inputs)))
        self.norm = tf.keras.layers.LayerNormalization(name=f"{prefix}_norm3")
        # add & norm
        self.outputs = self.norm(self.add([self.inputs, self.ffout]))
        # create model and return (also store)
        # MARK DEV: CALL SUBSTITUTE
        self.model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs, name=f"{prefix}_ff")

    def __call__(self, inputs):
        return self.model(inputs)

class Encoder(TransformerBase):
    
    '''
        Transformer encoder (model assembly) class.
    '''
    def __init__(self, input_shape, key_dim, ff_dim, dropout=0.1, clipvalue=0.15, prefix="enc", **kwargs):
        super().__init__()
        self.input_tensor = tf.keras.layers.Input(shape=input_shape, dtype='float32', name=f"{prefix}_in0")
        # Self Attention Block
        self.attention_block = SelfAttention(input_shape, prefix=prefix, key_dim=key_dim,**kwargs)
        self.attention_output = self.attention_block(self.input_tensor)
        # attention_dropout = tf.keras.layers.Dropout(rate=dropout)(self.attention_output)
        self.attention_add = tf.keras.layers.Add()([self.input_tensor, self.attention_output])  # Skip connection
        self.attention_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(self.attention_add)
        # Feed Forward Block
        self.ff_block = FeedForward(input_shape, key_dim, ff_dim, dropout, prefix)
        self.ff_output = self.ff_block(self.attention_norm)
        # self.ff_dropout = tf.keras.layers.Dropout(rate=dropout)(self.ff_output)
        # add & norm
        self.end_sum_norm = tf.keras.layers.Add()([self.attention_norm, self.ff_output])  # Skip connection
        self.encoder_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(self.end_sum_norm)
        # MARK DEV: CALL SUBSTITUTE
        self.model = tf.keras.Model(inputs=self.input_tensor, outputs=self.encoder_output, name=f"{prefix}_encoder")

    def __call__(self, inputs):
        return self.model(inputs)
    

class TransformerClassifier(TransformerBase):
    '''
        Full transformer encoder classification (model assembly) class.
        MARK DEV: dict of different dropouts; leave 
    '''
    def __init__(self, num_layers, 
                num_heads=2,
                seq_length=640, 
                key_dim=128, 
                ff_dim=256,
                vocab_size=10000,
                dropout=0.1, 
                last_dense=32,
                from_logits=False, 
                name="transformer_classifier"):
        super().__init__()
                  
        # output shape of positional embedding layer
        self.embed_shape = (seq_length, key_dim) 

        # input layer
        self.input_enc = tf.keras.layers.Input(shape=(seq_length,),
                                            dtype='int32',
                                            name='encoder_inputs')

        # for now: use trainable embeddings for both position & tokens
        # self.embed_enc = TokenPositionEmbeddingOriginal(seq_length, vocab_size, key_dim)

        self.embed_enc = TokenPositionEmbeddingLookup(seq_length, vocab_size, key_dim)

        # create as many sequential encoders as is called for
        encoders = [Encoder(input_shape=self.embed_shape,
                            key_dim = key_dim,
                            ff_dim = ff_dim,
                            dropout=dropout,
                            prefix=f"enc{i}",
                            num_heads=num_heads)
                    for i in range(num_layers)]

        self.final1 = tf.keras.layers.GlobalAveragePooling1D(name="transformer_finalFF_globAvgPool")
        # dropout here not currently implemented 
        self.drop = tf.keras.layers.Dropout(rate=dropout, name="transformer_finalFF_dropout")
        # final dense layer before output
        self.final2 = tf.keras.layers.Dense(last_dense, activation='relu', name="transformer_finalFF_finalDense")#, kernel_regularizer=tf.keras.regularizers.l2(0.01))

        
        # output layer depending on which computational formulation of binary cross-entropy we would like
        if from_logits:
            # we want logits for numerical stability reasons if from_logits=True
            # ...which linear activation gives us in our final layer
            self.final3 = tf.keras.layers.Dense(1, activation='linear', name="transformer_finalFF_modelOut")
        else:
            # we want normalized probabilities
            # ... which sigmoid activation gives us in our final layer
            self.final3 = tf.keras.layers.Dense(1, activation='sigmoid', name="transformer_finalFF_modelOut")
        # arrange model in order
        x1 = self.embed_enc(self.input_enc)
        for layer in encoders:
            x1 = layer(x1)
            self.outputs = self.final1(x1)
            self.outputs = self.drop(self.outputs)
            self.outputs = self.final2(self.outputs)
            self.outputs = self.drop(self.outputs)
            self.outputs = self.final3(self.outputs)
            try:
                del self.outputs._keras_mask
            except AttributeError:
                pass  
        # build model in prescribed order
        # MARK DEV: CALL SUBSTITUTE
        self.model = tf.keras.Model(inputs=[self.input_enc],
                                outputs=self.outputs, name=name)
    
    def __call__(self, inputs):
        return self.model(inputs)

