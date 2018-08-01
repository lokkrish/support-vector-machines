function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_poss = [0.01;0.03;0.1;0.3;1;3;10;30];
sig_poss = [0.01;0.03;0.1;0.3;1;3;10;30];
error_vec = zeros(length(C_poss),length(sig_poss));
for i = 1:length(C_poss);
   for j = 1:length(C_poss);
    sigma = sig_poss(j);
    model = svmTrain(X, y, C_poss(i), @(x1, x2) gaussianKernel(x1, x2, sigma));
    visualizeBoundary(X, y, model);
    prediction = svmPredict(model, Xval);
    error_vec(i,j) = mean(double(prediction ~=yval));
  endfor
endfor
[val, sigma_pos] = min(error_vec(:));
[rd, cd] = ind2sub(size(error_vec), sigma_pos);
C = C_poss(rd);
sigma = sig_poss(cd);

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
%







% =========================================================================

end
