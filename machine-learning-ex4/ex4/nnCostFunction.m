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

%% here for forward propagation
X = [ones(m,1) X];  % add one column as all 1
y1 = zeros(m,num_labels);  % 5000*10
for i=1:m
   y1(i,y(i)) = 1; 
end
% the y1 is the new y matrix with m*num_labels  5000*10
z2 = X*(Theta1.');
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*(Theta2.');
a3 = sigmoid(z3);   % should be 5000*10 dim
theta_1 = Theta1(:,2:end).^2;
theta_2 = Theta2(:,2:end).^2;
part1 = -y1.*log(a3);
part2 = (y1-ones(size(a3))).*log(ones(size(a3))-a3);
J = (sum(part1(:))+sum(part2(:))+0.5*lambda*(sum(theta_1(:))+sum(theta_2(:))))/m;  % with regularization


%% here compute the grad for backpropagation
Theta1_grad = zeros(size(Theta1));  % dim 25*401
Theta2_grad = zeros(size(Theta2));  % dim 10*26
for i=1:m
   %step 1
   a1 = X(i,:);  % dim 1*401
   z2 = a1*(Theta1.');  %  dim 1*25
   a2 = sigmoid(z2);
   a2 = [1 a2];  %dim 1*26
   z3 = a2*(Theta2.');  % dim 1*10
   a3 = sigmoid(z3);  % dim 1*10
   delta_3 = (a3 - y1(i,:)).';  % dim 10*1
   delta_2 = ((Theta2.')*delta_3); % dim 26*1
   delta_2 = delta_2(2:end).*sigmoidGradient(z2(:)); % dim 25*1
   Delta_theta1 = delta_2*a1;    %dim 25*401
   Delta_theta2 = delta_3*a2;  % dim 10*26
   Theta1_grad = Theta1_grad + Delta_theta1;
   Theta2_grad = Theta2_grad + Delta_theta2;
  
end

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = (Theta1_grad+lambda*Theta1)/m;
Theta2_grad = (Theta2_grad+lambda*Theta2)/m;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
