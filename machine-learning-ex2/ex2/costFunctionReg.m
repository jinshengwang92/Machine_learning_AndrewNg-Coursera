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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
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
theta(1) = 0;  % first theta0 will not be counted
J = (temp_J+lambda/2.0*sum(theta.^2))/m;
grad = (temp_grad+1.0*lambda*theta)/m;





% =============================================================

end
