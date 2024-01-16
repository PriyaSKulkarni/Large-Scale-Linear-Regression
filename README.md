# Large-Scale-Linear-Regression
CSE 6332 Cloud Computing and Big Data

Linear Regression Implementation

Implemented closed-form solution for linear regression with a small dataset:

Implementation Steps:

Created an RDD for matrix X and vector y.
Computed  using the outer product method.
Converted the result matrix to a Breeze Dense Matrix and computed the inverse.

Bonus (30 pts): Gradient Descent Update

Implemented gradient descent update for linear regression:

Implementation Steps:

Initialized 
0.001
α=0.001.
Implemented a function to compute the summand  and tested it on two examples.
Implemented a function to compute Root Mean Squared Error (RMSE) for an RDD of (label, prediction) tuples.
Tested the RMSE function on an example RDD.
Implemented a function for gradient descent updates, returning weights and training errors.
Tested the gradient descent function on an example RDD.
Ran the function for 5 iterations and printed the results.

Results:
Provided effective solutions for linear regression with both closed-form and gradient descent approaches. Demonstrated proficiency in distributed computing with RDDs and optimization algorithms for machine learning models.
