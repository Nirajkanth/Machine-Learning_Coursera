function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
%disp(size(yval));

randomvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
t = length(randomvalues);
list = zeros(t^2,3);
for i = 1:length(randomvalues),
  for j= 1:t,
    k = j;
    j = t*(i-1) + j;
    list(j,1) = randomvalues(i);
    list(j,2) = randomvalues(k);
    model= svmTrain(X, y, list(j,1), @(x1, x2) gaussianKernel(x1, x2, list(j,2)));   
    predictions = svmPredict(model, Xval);
    list(j,3) = mean(double(predictions ~= yval));
  end
end

%
disp(list);
%perror = zeros(t^2,1);
%for i = 1:t^2,
%    C_t = list(i,1);
%    sigma_t = list(i,2);
%    model= svmTrain(X, y, C_t, @(x1, x2) gaussianKernel(x1, x2, sigma_t)); 
%    predictions = svmPredict(model, Xval);
%    perror = mean(double(predictions ~= yval));
%end


shor_re = sortrows(list,3);


C= shor_re(1,1);
sigma = shor_re(1,2);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
% =========================================================================

end
