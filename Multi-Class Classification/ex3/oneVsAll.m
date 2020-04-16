function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 

for k = 1:num_labels
    initial_theta = zeros(n + 1, 1); % initial guess
    % setting options for fmincg optimization function
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    % retrieving optimal theta vector for the kth logistic regression
    [optTheta] = fmincg (@(t)(lrCostFunction(t, X, (y == k), lambda)), ...
                    initial_theta, options);
    % set optTheta to row k in the all_theta matrix
    all_theta(k, :) = optTheta;
end
% =========================================================================
end
