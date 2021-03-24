function [J grad] = nnCostFunction(nn_params, ...
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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
X = [ones(m,1) X];
a2 = sigmoid(X*Theta1');
a2 = [ones(m,1),a2];
a3 = sigmoid(a2*Theta2');
h = a3;

vec_y = zeros(m,num_labels);
for i =1:m
  vec_y(i,y(i)) = 1;
end 
% vet_y = (1:num_labels)==y
temp1 = Theta1;
temp2 = Theta2;
temp1(:,1) = 0;
temp2(:,1) = 0;
reg = (lambda/(2*m))*((sum(sum(temp1.^2))) + sum(sum(temp2.^2)));
J = (-1/m)*sum(sum(vec_y.*log(h) + (1- vec_y).*log(1-h))) + reg;

Theta1_grad = zeros(size(Theta1)); % 25x401
Theta2_grad = zeros(size(Theta2)); %10x26
for t = 1:m
  a1 = X(t,:);   % 1x401
  a2 = sigmoid(a1*Theta1');  % 1x25
  a2 = [1 a2];   %1x26
  a3 = sigmoid(a2*Theta2');  %1x10
  
  % y = 1x1
  
  vect_y = (1:num_labels)== y(t); % 1x10
  Delta3 = (a3 - vect_y)';    % 10x1
  Delta2 = ((Theta2'([2:end],:)*Delta3).*(sigmoidGradient(a1*Theta1'))');  % t2*d3 = (25x10)*(10x1)  , s = 25x1
  %Delta2 = ((Theta2'([2:end],:)*Delta3).*(sigmoidGradient(a1*Theta1'))');
  
  Theta1_grad = Theta1_grad + Delta2*a1 ;  % da = 25x1 1x401
  Theta2_grad = Theta2_grad + Delta3*a2 ;  
end
temp1 = Theta1;
temp2 = Theta2;
temp1(:,1) = 0;
temp2(:,1) = 0;
Theta1_grad = (1/m)*Theta1_grad  + (lambda/m)*(temp1);
Theta2_grad = (1/m)*Theta2_grad  + (lambda/m)*(temp2);
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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
