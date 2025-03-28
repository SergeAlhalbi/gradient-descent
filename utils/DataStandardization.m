function [y, mu, sigma] = DataStandardization(x)
%
% y = DataStandardization(x)
%
% DataStandardization standardizes the data by shifting it towards the center and scaling it to one standard deviation difference.
%
% INPUTS:
%
%   x - Input original data
%       Type: single, double, etc.
%
% OUTPUTS:
%
%   y - Output standardized data
%       Type: single, double, etc.
%
%   mu - Mean of the input data
%        Type: single, double, etc.
%
%   sigma - Standard deviation of the input data
%           Type: single, double, etc.

%% Standardizing the data
% Shifting
mu = mean(x);
y = x - mu;

% Scaling
sigma = std(y);
y = y ./ sigma;

return