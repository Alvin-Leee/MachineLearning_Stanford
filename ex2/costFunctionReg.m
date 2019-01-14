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

h = sigmoid(X*theta);
[p, f] = costFunction(theta, X, y);
rest = sum(theta.*theta)-theta(1)*theta(1);
J = p+rest*lambda/(2*m);

%disp(grad);
%grad = sum(X'*(h - y))/m+lambda.*theta/m;
%grad(1) = sum((h-y).*X(:, 1))/m;
%disp(grad);

for j = 1:size(theta)
    if j==1
        grad(j) = sum((h-y).*X(:, 1))/m;
    else
        grad(j) = sum((h-y).*X(:, j))/m + lambda .* theta(j)/m;
    end
end
%disp(grad);





% =============================================================

end
