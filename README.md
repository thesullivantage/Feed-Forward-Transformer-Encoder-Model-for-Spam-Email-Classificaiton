# Feed-Forward Transformer Encoder Model for Spam Email Classification

- A deep, feed-forward classification model built using text vectorization, custom pre-processing using the Pandas library in Python, and a transformer encoder network built in Keras for spam email classification.

- The client initially followed [this tutorial](https://machinelearningmastery.com/building-transformer-models-with-attention-crash-course-build-a-neural-machine-translator-in-12-days/) in the construction of this model.
  - __Transformer model was adapted for the text classification problem at hand, rather than the translation task described in the tutorial.__
  - __I re-wrote the feature engineering code using Pandas in addition to modularizing model code for easier conversion to PyTorch later.__
  - __Following refactoring in PyTorch, deployment would entail the construction of an AWS Sagemaker pipeline for training, evaluation, and serving of inferences in a client application. This will also serve adaptation of the model code to other NLP and computer vision problems, alike. This is out of scope for this contract.__

- Data is excluded here but can be found on [this Kaggle page](https://www.kaggle.com/datasets/ozlerhakan/spam-or-not-spam-dataset). Any labeled spam email dataset with a structure of ['email_text', 'label'], containing a reasonable distribution of word counts per 'email_text' example, should be usable with the model. Some feature-engineering adjustment may be required.

