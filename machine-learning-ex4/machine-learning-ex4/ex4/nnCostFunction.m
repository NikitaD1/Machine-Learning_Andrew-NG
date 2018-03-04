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
X=[ ones(m,1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
   z2=Theta1*X';
   htheta = sigmoid(Theta1*X');
   htheta=[ ones(1,m) ; htheta];
   h3 = sigmoid(Theta2*htheta);
  
   
   z= zeros(m,num_labels);
   for (i=1: m)
      
       z(i,y(i))=1;
       
   end
   
   S = sum( -z'.*log(h3) -  (1-z)'.*log(1- h3),1);
  J= 1/m*sum( S,2);
  % after regularisation
  Theta11 =Theta1(:,2:size(Theta1,2));
  Theta22 =Theta2(:,2:size(Theta2,2));
  J = J + lambda/2/m*(sum(sum(Theta22.^2,1),2) + sum(sum(Theta11.^2,1),2));
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
h3t=h3';
DEL2=0;
DEL1=0;
%size(htheta)
z2 = [ones(1,size(z2,2)) ; z2];    
for t=1:m
    
    %a1=X(1,t)';
    delta3 = (-z(t,:)+h3t(t,:))' ;
    %size(delta3)
    
    %size(z2)
    %size(Theta2)
    %size(delta3)
    %size(z2(:,t))
    delta2 = (Theta2'*delta3).* sigmoidGradient(z2(:,t));
    delta2 = delta2(2:end)';
   % size(delta2)
    %size(X(t,:)')
    DEL2 = DEL2 + delta3*htheta(:,t)';
    DEL1 = DEL1 + delta2'*X(t,:) ;
end
%grad= DEL1/m;
%size(DEL2)
%size(DEL1)
Theta2_grad=DEL2/m;
Theta1_grad=DEL1/m;

Theta2_grad = Theta2_grad + ...
              (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + ...
              (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);
grad = [Theta1_grad(:) ; Theta2_grad(:)];

    
    

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
