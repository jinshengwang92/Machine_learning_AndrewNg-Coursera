function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
temp_J = 0;
l_theta = length(theta);
temp_grad = zeros(size(theta));
for j=1:m
    h_theta = sigmoid(X(j,:)*theta);
    temp_J = temp_J + (-y(j)*log(h_theta)-(1-y(j))*log(1-h_theta));
    for k = 1:l_theta
        temp_grad(k) = temp_grad(k) + (h_theta-y(j))*X(j,k);
    end
end
J = temp_J/m;
grad = temp_grad/m;









% =============================================================

end
