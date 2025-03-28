function [ errMsg, varargout ] = CheckInputs( names, defaults, varargin )
%
% [ errMsg, varargout ] = CheckInputs( names, defaults, OPTIONS )
%
% Generic utility for assigning defaults to various inputs. This routine can be used by functions
% that accept variable number of input arguments.
%
% INPUTS:
%
%   names    - A cell array of strings defining the names of all possible input arguments
%
%   defaults - A cell array of default values, one for each variable name
%
% OPTIONAL INPUTS:
%
%   OPTIONS - An N*2 cell array, where N is the number of variable names and corresponding values
%             provided by the user. N must be less than or equal to length(names)/2
%
% OUTPUTS:
%
%   errMsg    - A string defining any encountered error
%
%   varargout - Value of all inputs (default or those provided by the user)
%

% Copyright (c) Riverain Technologies (2013-2018)

% Initialize outputs
errMsg = '';
varargout = defaults;

% Number of user specified inputs
numInputs = length( varargin );

% A quick check of the inputs
if ~iscell( names )
    errMsg = 'The list of variable names should be a cell array.';
    return
end
if ~iscell( defaults )
    errMsg = 'The list of default input values should be a cell array.';
    return
end
if length( names ) ~= length( defaults )
    errMsg = 'The lengths of the input names and defaults are inconsistent.';
    return
end
if rem( numInputs, 2 ) ~= 0
    errMsg = 'Every input does not have a corresponding value.';
    return
end

% Overwrite defaults with specified inputs
for k = 1 : 2 : numInputs - 1
    
    inputIdx = find( strcmpi( varargin{ k }, names ) );
    
    if isempty( inputIdx )
        errMsg = [ 'Input name "', varargin{ k }, '" is not recognized.' ];
        return
    else
        varargout{ inputIdx } = varargin{ k + 1 };
    end
    
end

return