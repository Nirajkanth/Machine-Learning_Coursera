 function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h = sigmoid(X*theta);
n = length(theta);
newTheta = theta([2:n],:);
J =((-1/m)*(y'*log(h) + (1-y)'*log(1-h)) + (lambda/(2*m))*(sum(newTheta.^2)));

subgrad = (1/m)*X'*(h-y);
gradzero = subgrad(1);
othergrad = (subgrad([2:length(subgrad)],:) + (lambda/m)*(newTheta));
grad =[gradzero;othergrad];

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
