import sys
import os
import pickle
# So we can use model_tutorial_pkg while not in the same working dir:
### REPLACE WITH YOUR PATH
sys.path.append(os.getcwd())

import model_tutorial_pkg as mod
import os 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# create training output directory
### MARK DEV: using argparse
train_dat_dir = 'justin_training_data'
if not os.path.exists(train_dat_dir):
    os.makedirs(train_dat_dir)
    

# load already pre-processed data (already in numpy ndarray format)
with open("../data/vectorizer_data/normalized_split_data.pickle", "rb") as fp:
    dataset_dict = pickle.load(fp)

X_train = dataset_dict['X_train']
y_train = dataset_dict['y_train']
X_test = dataset_dict['X_test']
X_valid = dataset_dict['X_valid'] 
y_test = dataset_dict['y_test'] 
y_valid = dataset_dict['y_valid']
X_train_vec = dataset_dict['X_train_vec']
X_test_vec = dataset_dict['X_test_vec']
X_valid_vec = dataset_dict['X_valid_vec']
print(X_train_vec.shape)


batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_vec, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_vec, y_test)).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid_vec, y_valid)).batch(batch_size)

##################### MODEL & TRAINING #####################

# pre-processed data and model parameters
seq_length = 640
embed_dim = 32
ff_dim = 64
num_transformer_layers = 1
num_heads = 2
key_dim = embed_dim
dropout = 0.2
vocab_size_en = 10000

# Adam optimizer parameters
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-7

# Use exponential decay schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-3,
    decay_steps=10000,
    decay_rate=0.9)

optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1 = beta_1, beta_2=beta_2, epsilon=epsilon, clipvalue=0.5)

# stored as variable for both bce_loss and model construction 
# if true, last layer has linear activations and loss instance applies sigmoid function. 
# if false, last layer has sigmoid activations and loss instance computes loss directly. 
from_logits=True

############### MODEL & LOSS CONFIGURATION ###############

# Some notes for Justin:
# I changed the transformer method signature (in model_tutorial_pkg.model_combined) to have more optional parameters (with "=" sign) 
# so we: 1) we don't have to pass args in a special order 2) have some default args to accomodate end-users down the line

# pass parameters in
model = mod.transformer(num_transformer_layers, 
                        num_heads=num_heads, 
                        seq_length=seq_length, 
                        key_dim=key_dim,
                        ff_dim=ff_dim, 
                        vocab_size=vocab_size_en, 
                        dropout=dropout,
                        from_logits=from_logits,
                        last_dense=20)

# compile model before training
model.compile(optimizer="adam", ### no custom LR schedule (from paper)
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),# reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
              metrics=["accuracy"])


# binary cross-entropy loss
bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

############### TRAINING LOOP ###############

# Initialize training history lists to store data every step
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []
y_preds = []
y_vals = []
y_vals_valid = [] 
y_val_preds = []
grad_history = []

def recordGrad(grads, model, gradhistory):
    data = {}
    for g,w in zip(grads, model.trainable_weights):
        if '/kernel:' not in w.name:
            continue # skip bias
        name = w.name.split("/")[0]
        data[name] = g.numpy()
    gradhistory.append(data)
    return

# Training loop parameters
epochs = 5
# Print loss every 'print_frequency' training steps
print_frequency = 1 

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    for step, (x_batch, y_batch) in enumerate(train_dataset):
        

        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            y_preds.append(y_pred)
            y_vals.append(y_batch)
            loss_value = bce_loss(y_batch, y_pred)  # apply loss function

        grads = tape.gradient(loss_value, model.trainable_variables)
        if step == 0:
            recordGrad(grads, model, grad_history)
            
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_history.append(loss_value.numpy())
        # Print loss and one output example
        if step % print_frequency == 0:
            print(f"Step {step}/{len(train_dataset)}, Loss: {loss_value.numpy()}")
            example_output = y_pred[0].numpy()  # Get the output of the first example
            print(f"Example Output: {example_output}")

    # Validation step (similar to training step, but without gradient computation)
    for x_val, y_val in valid_dataset:
        y_val_pred = model(x_val, training=False)
        y_val_preds.append(y_val_pred)
        y_vals_valid.append(y_val)
        val_loss = bce_loss(y_val, y_val_pred) 
        val_loss_history.append(val_loss.numpy())


### MARK DEV: move to train_profile.py script
### after training complete
recordGrad(grads, model, grad_history)

### MARK DEV: move to train_profile.py script
# Plot gradient mean and sd across epochs
fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(8, 12))
ax[0].set_title("Mean gradient")
for key in grad_history[0]:
    ax[0].plot(range(len(grad_history)), [w[key].mean() for w in grad_history], label=key)
ax[0].legend()
ax[1].set_title("S.D.")
for key in grad_history[0]:
    ax[1].semilogy(range(len(grad_history)), [w[key].std() for w in grad_history], label=key)
ax[1].legend()
ax[2].set_title("Loss")
ax[2].plot(range(len(train_loss_history)), train_loss_history)
fig.savefig(os.path.join(train_dat_dir, 'gradients_loss_plot.png'), dpi=300)


print('Saving model weights...')
model.save_weights(os.path.join(train_dat_dir,'justin_model_weights_altered.h5'))
print('Model weights saved.')

train_data = {
    'train_loss_history': train_loss_history,
    'train_accuracy_history': train_accuracy_history,
    'val_loss_history': val_loss_history,
    'val_accuracy_history': val_accuracy_history,
    'y_preds': y_preds,
    'y_val_preds': y_val_preds,
    'y_vals': y_vals,
    'y_vals_valid': y_vals_valid,
}

print('Saving training data...')
with open(os.path.join(train_dat_dir, 'training_data_1.pickle'), "wb") as fp:
    ### Save Un-Tensorized Datasets
    data = {
        'train_data': train_data
    }
    pickle.dump(data, fp)
print('Training data saved.')



