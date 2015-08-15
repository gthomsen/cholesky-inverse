function X_inverse = cholesky_inverse( X, precision )
% X_inverse = cholesky_inverse( X, precision )
%
% Computes the inverse of X using Cholesky decomposition.  The precision the
% inversion is performed in may be specified so as to reduce numerical error
% in the process.  If the supplied matrix is not positive definite, an error
% is thrown.
%
% Takes 2 arguments:
%
%   X         - Positive definite matrix to invert using Cholesky
%               decomposition.
%
%   precision - Optional string indicating the precision that the inversion
%               should performed in.  Must be one of the following:
%
%                 'double', 'float64' -  Double precision.
%                 'single', 'float32' -  Single precision.
%
%               If omitted, inversion is performed in the same precision as X.
%
%               NOTE: This only affects the precision used for inversion and
%                     not the precision of the inverse returned.
%
% Returns 1 value:
%
%   X_inverse - Inverse of X.

return
