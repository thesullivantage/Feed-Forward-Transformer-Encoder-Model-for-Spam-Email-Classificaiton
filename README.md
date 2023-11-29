# Feed-Forward Transformer Encoder Model for Spam Email Classification

- A deep, feed-forward classification model built using text vectorization, custom pre-processing using the Pandas library in Python, and a transformer encoder network built in Keras for spam email classification. 

- Designed for an independent client. Future work will entail refactoring the model in PyTorch and the construction of an AWS Sagemaker pipeline for training, evaluation, and serving of inferences.

- Data is excluded here. Any labeled spam email dataset with a structure of ['email_text', 'label'], containing a reasonable distribution of word counts per 'email_text' example, should be usable with the model. Some adjustment may be required.
