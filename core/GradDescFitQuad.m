function theta = GradDescFitQuad(fhndl, x, y, varargin)
%
% theta = GradDescFitQuad(fhndl, x, y, OPTIONS)
%
% GradDescFitQuad performs a gradient descent to find the optimized set of parameters.
%
% INPUTS:
%
%   fhndl - Function handle used by feval to get the function value, the
%           error, and the error gradient
%           Type: function handle
%
%   x - Domain of the function
%       Type: single, double, etc.
%
%   y - Measured output data
%       Type: single, double, etc.
%
% OPTIONS (string-value pairs):
%
%   'theta0' - Vector containing the initial parameters
%              DEFAULT: uniformly random value for the weight between -1 and 1 and 0 for the bias
%              Type: single, double, etc.
%
%   'lr' - Learning rate
%          DEFAULT: 0.1; Type: single, double, etc.
%
%   'maxIters' - Maximum number of iterations
%                DEFAULT: 50; Type: single, double, etc.
%
%   'verbose' - A binary flag that controls whether the error is printed
%               DEFAULT: false -> No printing of progress
%
%   alpha - Lasso regularizer term
%           DEFAULT: 0; Type: single, double, etc.
%
%   beta - Ridge regularizer term
%           DEFAULT: 0; Type: single, double, etc.
%
% OUTPUTS:
%
%   theta - Vector containing the optimized set of parameters
%           Type: single, double, etc.

%% Setting defaults
% Setting the options
names = {'theta0', 'lr', 'maxIters', 'verbose', 'alpha', 'beta'};
defaults = {[2*rand(1) - 1; 2*rand(1) - 1; 0], 0.1, 50, false, 0, 0};
[errMsg, theta0, lr, maxIters, verbose, alpha, beta] = CheckInputs(names, defaults, varargin{:});
error(errMsg);

% Defining the first output set of parameters as the initial one
theta = theta0;

%% Gradient descent

for iter = 1:maxIters

    % Forward and backward passing
    [yp, errVal, errGradVec] = feval(fhndl, theta, x, y, 'alpha', alpha, 'beta', beta);

    % Updating the paramaters (gradient descent)
    theta = theta - lr .* errGradVec;

    % Print progress
    if verbose

        fprintf(1,'Iteration %3d:   Error: %5.15f\n', iter, errVal);

    end

end

return