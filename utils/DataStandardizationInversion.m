function y = DataStandardizationInversion(x, mu, sigma)
%
% y = DataStandardizationInversion(x)
%
% DataStandardizationInversion gets back the original unstandardized data.
%
% INPUTS:
%
%   x - Input standardized data
%       Type: single, double, etc.
%
%   mu - Mean of the original data
%        Type: single, double, etc.
%
%   sigma - Standard deviation of the original data
%           Type: single, double, etc.
%
% OUTPUTS:
%
%   y - Output original data
%       Type: single, double, etc.

%% Unstandardizing back the data
% Scaling back
y = x .* sigma;

% Shifting back
y = y + mu;

return