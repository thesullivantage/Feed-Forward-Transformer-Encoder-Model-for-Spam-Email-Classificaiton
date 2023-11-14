import os 
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import auc

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

### LOAD (already pre-proc. data)
with open("../../data/vectorizer_data/normalized_split_data.pickle", "rb") as fp:
    dataset_dict = pickle.load(fp)

# data pre-processed and vectorized using as numpy
X_train = np.array(dataset_dict['X_train'])
y_train = np.array(dataset_dict['y_train'])
X_test = np.array(dataset_dict['X_test'])
X_valid = np.array(dataset_dict['X_valid'] )
y_test = np.array(dataset_dict['y_test'] )
y_valid = np.array(dataset_dict['y_valid'])
X_train_vec = np.array(dataset_dict['X_train_vec'])
X_test_vec = np.array(dataset_dict['X_test_vec'])
X_valid_vec = np.array(dataset_dict['X_valid_vec'])

y_concat = np.concatenate((y_train, y_test, y_valid))

### The code below plots histograms of label counts in each of our three datasets (train, test, validation).
### We would like to see this at the feature engineering step (before training), but I do it here, since I did it in print statements
# This set (x-lim) edges of bins
bins = [0, 1, 2]  

plt.hist(y_concat, bins=bins, alpha=0.7, rwidth=0.85, edgecolor='black', density=True)
plt.title('Normalized Histogram of\n(All) Data Labels')
plt.xlabel('Label Values')
plt.ylabel('Counts')
plt.xticks([0.5, 1.5], ['0', '1'])  # Set the x-ticks to be the center of the bins
plt.grid(axis='y', alpha=0.75)
plt.savefig('justin_model_plots/counts_all_data.png')
plt.close('all')
# Training Data
plt.hist(y_train, bins=bins, alpha=0.7, rwidth=0.85, edgecolor='black', density=True)
plt.title('Normalized Histogram of\nTraining Data Labels')
plt.xlabel('Label Values')
plt.ylabel('Counts')
plt.xticks([0.5, 1.5], ['0', '1'])  # Set the x-ticks to be the center of the bins
plt.grid(axis='y', alpha=0.75)
plt.savefig('justin_model_plots/counts_train_data.png')
plt.close('all')
# Validation Data
plt.hist(y_valid, bins=bins, alpha=0.7, rwidth=0.85, edgecolor='black', density=True)
plt.title('Normalized Histogram of\nValidation Data Labels')
plt.xlabel('Label Values')
plt.ylabel('Counts')
plt.xticks([0.5, 1.5], ['0', '1'])  # Set the x-ticks to be the center of the bins
plt.grid(axis='y', alpha=0.75)
plt.savefig('justin_model_plots/counts_valid_data.png')
plt.close('all')

# read-in training data
train_data_path = '../justin_training_data/training_data_1.pickle'
with open(train_data_path, "rb") as fp:
    train_data = pickle.load(fp)
    
# get training loss history
loss_hist = np.array(train_data['train_data']['train_loss_history'])
# concatenate predictions (floats) from training and validation datasets
y_train_preds = np.concatenate(train_data['train_data']['y_preds'])
y_val_preds = np.concatenate(train_data['train_data']['y_val_preds']) ## validation data predictions
# compute probabilities
# this is for from_logits=True case, where the model outputs (our preds) haven't been passed through a sigmoid activation yet
# so we do so below
y_train_probs = sigmoid(y_train_preds)
y_val_probs = sigmoid(y_val_preds)

# get validation loss history
loss_val_hist = np.array(train_data['train_data']['val_loss_history'])

# concatenate labels (ints) from training and validation datasets
y_train_values = train_data['train_data']['y_vals']
y_train_values = np.concatenate([y.numpy() for y in y_train_values])
y_valid_values = train_data['train_data']['y_vals_valid']
y_valid_values = np.concatenate([y.numpy() for y in y_valid_values])

# precision_train, recall_train, thresholds_train = precision_recall_curve(y_train_values, y_train_probs)
precision_valid, recall_valid, thresholds_valid = precision_recall_curve(y_valid_values, y_val_probs)

# calculate AUC (area under curve) of precision-recall plot
auc_PR_valid = round(auc(recall_valid, precision_valid), 2)

# probability of randomly drawing a positive sample: corresponding to a "no-skill" model
no_skill_frac_valid = round(len(y_valid_values[y_valid_values==1]) / len(y_valid_values), 2)

# plot precision-recall curve of our classifier against our 
plt.plot(recall_valid, precision_valid, 'r--', label=f'AUC={auc_PR_valid}')
plt.plot([0, 1], [no_skill_frac_valid, no_skill_frac_valid], linestyle='--', label=f'No Skill AUC={no_skill_frac_valid}')
plt.title('Precision-Recall: Validation Data')
plt.xlabel('Recall Values')
plt.ylabel('Precision Values')
plt.grid(axis='y', alpha=0.75)
plt.legend()
plt.savefig('justin_model_plots/prec_recall_valid.png', dpi=300)
plt.close('all')