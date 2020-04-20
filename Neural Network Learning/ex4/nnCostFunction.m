function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% Return values 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Delta_2 = 0;
Delta_1 = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------- MULTI-CLASS OUTPUT --------- %
eye_matrix = eye(num_labels);
Y = eye_matrix(y,:); % 5000 x 10

% -------- FORWARD PROPAGATION -------- %
a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

% -------- COST REGULARIZATION -------- %
% Don't regularize bias term
Theta1_r = [zeros(size(Theta1, 1), 1) Theta1(:, 2:end)]; % (25 x 400)
Theta2_r = [zeros(size(Theta2, 1), 1) Theta2(:, 2:end)]; % (10 x 25)
reg_term = ((sum(sum(Theta1_r .* Theta1_r)) + sum(sum(Theta2_r .* Theta2_r))) ...
            * lambda) / (2 * m);

% ----------- COMPUTE COST ------------ %
J = - (sum(sum(Y.*log(a3))) + sum(sum((1 - Y).*log(1 - a3)))) / m + reg_term;

% --------- BACK PROPAGATION ---------- %
% Calculate sigmas
sigma3 = a3 - Y;
sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]);
sigma2 = sigma2(:, 2:end);

% Accumulate gradients
Delta_1 = (sigma2'*a1);
Delta_2 = (sigma3'*a2);

% -------- GRAD REGULARIZATION -------- %
% Don't regularize bias term
p1 = (lambda * Theta1_r) / m;
p2 = (lambda * Theta2_r) / m;

% Get derivatives
Theta1_grad = Delta_1./m + p1;
Theta2_grad = Delta_2./m + p2;

% ----------- UNROLL GRADIENTS--------- %
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
