function [x, y, t] = CreateGradDescLinRegData(noiseVariance, numberOfPoints, xLeft, xRight, theta, varargin)
%
% [x, y, t] = CreateGradDescLinRegData(noiseVariance, numberOfPoints, xLeft, xRight, theta, OPTIONS)
%
% CreateGradDescLinRegData generates data using a linear or a quadratic function,
% and a Gaussian noise for linear regression.
%
% INPUTS:
%
%   fhndl - Function handle used by feval to get the function value
%           Type: function handle
%
%   noiseVariance - Variance of the Gaussian noise added to the actual output data
%                   Type: single, double, etc.
%
%   numberOfPoints - Number of points in the data
%                    Type: single, double, etc.
%
%   xLeft, xRight - Smallest and largest input values, respectively
%                   Type: single, double, etc.
%
%   theta - Vector containing the parameters
%           Type: single, double, etc.
%
% OPTIONS (string-value pairs):
%
%   'seedVal' - Seed value for repeatable results
%               DEFAULT: 1; Type: single, double, etc.
%
% OUTPUTS:
%
%   x - Input data
%       Type: single, double, etc.
%
%   y - Measured output data
%       Type: single, double, etc.
%
%   t - True target output data
%          Type: single, double, etc.

%% Setting defaults
names = {'seedVal'};
defaults = {1};
[errMsg, seedVal] = CheckInputs(names, defaults, varargin{:});
error(errMsg);

%% Defining and computing the parameters
% Fixing a seed for repeatable results
rng(seedVal)

% Defining the function input
x = linspace(xLeft, xRight, numberOfPoints);

% Evaluating the function
if length(theta) == 2

    t = theta(1)*x + theta(2);

else

    t = theta(1)*(x.^2) + theta(2)*x + theta(3);
    
end

% Adding Gaussian noise to the actual output data
y = t + noiseVariance*randn(size(x));

return