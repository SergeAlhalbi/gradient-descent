function y = OutliersAddition(x, numOutliers, varargin)
%
% y = OutliersAddition(x, numOutliers, OPTIONS)
%
% OutliersAddition adds outliers to the data
%
% INPUTS:
%
%   x - Original data
%       Type: single, double, etc.
%
%   numOutliers - Number of outliers
%                 Type: single, double, etc.
%
% OPTIONS:
%
%   maxDiff - Maximum difference between the original and the new data
%             DEFAULT: 6; Type: single, double, etc.
%
% OUTPUTS:
%
%   y - New data with outliers
%       Type: single, double, etc.

%% For repeatable results
rng(1);

%% Setting defaults
% Setting the options
names = {'maxDiff'};
defaults = {6};
[errMsg, maxDiff] = CheckInputs(names, defaults, varargin{:});
error(errMsg);

%% Adding outliers
y = x;
indicesOutliers = randperm(length(y));
randIndicesOutliers = indicesOutliers(1:numOutliers);
y(randIndicesOutliers) = y(randIndicesOutliers) + (-1).^(rand(1, length(randIndicesOutliers))>0.5) .* (maxDiff .* rand(1, length(randIndicesOutliers)));

return