function [yp, errVal, errGradVec] = RidgeLossQuad(theta, x, y, varargin)
%
% [yp, errVal, errGradVec] = RidgeLossQuad(theta, x, y, OPTIONS)
%
% RidgeLossQuad computes the forward prediction using the current set of parameters,
% and the mean squared error and its gradient if demanded.
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
yp = theta(1)*(x.^2) + theta(2)*x + theta(3);

if nargin == 2
    errVal = []; errGradVec = [];
    return
end

%% If demanded: Computing the error and its gradient
% Finding the number of points
numberOfPoints = length(x);

% Computing the error (loss/cost/error function or normalized L2 norm)
errVal = (1/numberOfPoints) * (sum((y - yp).^2));

% Computing the gradient
errGradVec = (-2/numberOfPoints) .* [sum((y - yp).*(x.^2)); sum((y - yp).*(x)); sum(y - yp)];

if alpha ~= 0

    % Adding a Lasso regularizer to the error and the gradient
    errVal = errVal + (1/numberOfPoints) * (alpha*sum(abs(theta)));
    reguGradVecAlpha = (-2/numberOfPoints) .* alpha*(sign(theta));
    errGradVec = errGradVec + reguGradVecAlpha;

elseif beta ~= 0

    % Adding a Ridge regularizer to the error and the gradient
    errVal = errVal + (1/numberOfPoints) * (beta*sum(theta.^2));
    reguGradVecBeta = (-2/numberOfPoints) .* beta*(2*theta);
    errGradVec = errGradVec + reguGradVecBeta;

end

return