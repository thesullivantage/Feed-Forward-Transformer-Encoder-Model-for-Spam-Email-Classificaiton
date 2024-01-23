# Development Notes on MLP-DE Model - For Client

## Model Notes
- Model written in Python via the Keras subclassing API.
- Adjusted the dimension of inputs to: `(batch_size, number_of_features)`, since this is what Keras input layers expect.

## Loss Notes
- Linear formulation of neural network solution, including initial condition, makes sense. 
- The loss (objective) function relying mainly upon the numerical calculation of function derivative increases the complexity of this challenge.
- TODO: Epsilon (difference in forward difference numerical calculation) should probably be passed as a parameter. 

## Training Notes - Preliminary
- Training needs to take place on the unit-standardized interval (-1, 1) for stability of gradient descent.
- Up to this point, I have been implementing SGD: runninng gradient descent for every sample (batch size: 1).
- Training is currently not converging. __ Squared error values (using SGD; not computing mean) values appear consistent with the x-values themselves.__.
- Some thoughts on that:
    - Epsilon may be too small.
      - From the [numpy docs](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html): it is currently the 64-bit "difference between 1.0 and the next smallest representable float larger than 1.0."
      - But, this did not seem to affect the results of one other analyst who's tutorial I read through, which is where I lifted the loss implementation from.
  - For modifying optimization strategy - could try:
    - Adam optimization using a batch strategy (gradient descent run after one full pass through data - per epoch).
    - Modified SGD using a minibatches of data __to get MSE instead of individual squared error values__.
