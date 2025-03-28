function [yp, errVal, errGradVec] = LassoLossLine(theta, x, y, varargin)
%
% [yp, errVal, errGradVec] = LassoLossLine(theta, x, y, OPTIONS)
%
% LassoLossLine computes the forward prediction using the current set of parameters,
% and the absolute error and its gradient if demanded.
%
% INPUTS:
%
%   theta - Vector containing the parameters
%           Type: single, double, etc.
%
%   x - Domain of the function
%       Type: single, double, etc.
%
%   y - Actual output data
%       Type: single, double, etc.
%
% OPTIONS:
%
%   alpha - Lasso regularizer term
%           DEFAULT: 0; Type: single, double, etc.
%
%   beta - Ridge regularizer term
%           DEFAULT: 0; Type: single, double, etc.
%
% OUTPUTS:
%
%   yp - Predicted output data using the current set of parameters, theta
%        Type: single, double, etc.
%
%   errVal - Mean squared error
%            Type: single, double, etc.
%
%   errGradVec - Error gradient
%                Type: single, double, etc.

%% Setting defaults
% Setting the options
names = {'alpha', 'beta'};
defaults = {0, 0};
[errMsg, alpha, beta] = CheckInputs(names, defaults, varargin{:});
error(errMsg);

%% Evaluating the function at x
yp = theta(1)*x + theta(2);

if nargin == 2
    errVal = []; errGradVec = [];
    return
end

%% If demanded: Computing the error and its gradient
% Finding the number of points
numberOfPoints = length(x);

% Computing the error (loss/cost/error function or normalized L1 norm)
errVal = (1/numberOfPoints) * (sum(abs(y - yp)));

% Computing the gradient
errGradVec = (-1/numberOfPoints) .* [sum(sign(y - yp).*x); sum(sign(y - yp))];

if alpha ~= 0

    % Adding a Lasso regularizer to the error and the gradient
    errVal = errVal + (1/numberOfPoints) * (alpha*sum(abs(theta)));
    reguGradVecAlpha = (-1/numberOfPoints) .* alpha*(sign(theta));
    errGradVec = errGradVec + reguGradVecAlpha;

elseif beta ~= 0

    % Adding a Ridge regularizer to the error and the gradient
    errVal = errVal + (1/numberOfPoints) * (beta*sum(theta.^2));
    reguGradVecBeta = (-1/numberOfPoints) .* beta*(2*theta);
    errGradVec = errGradVec + reguGradVecBeta;

end

return