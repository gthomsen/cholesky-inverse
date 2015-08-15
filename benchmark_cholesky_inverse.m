function timings = benchmark_cholesky_inverse( N, number_iterations )
% timings = benchmark_cholesky_inverse( N, number_iterations )
%
% Benchmarks the Cholesky inverse MEX object against other alternatives for
% computing the explicit inverse within MATLAB and Octave.  Each of the
% following inversion techniques are run a number of times and the median
% time for each technique is recorded:
%
%   1. Inversion via inv().
%   2. Inversion via heuristic derived factorization and back substitution.
%   3. Inversion via Cholesky factorization and back substitution.
%   4. Inversion via the cholesky_inverse() MEX object.
%
% Timings are taken for single and double precision inputs, as well as
% Hermitian and symmetric matrices, resulting in four sets of timings.
%
% Takes 2 arguments:
%
%   N                 - The size of the matrix to use for benchmarks.  If omitted,
%                       defaults to 100.
%   number_iterations - The number of iterations to use during benchmarking.  If
%                       omitted, defaults to 20.
%
% Returns 1 value:
%
%   timings - Matrix, sized N x 4, of median timings in seconds.  Columns one
%             and two represents times for each of the N inversion techniques
%             using double precision data for Hermitian and symmetric
%             matrices, respectively, while the third and fourth columns
%             represents timings for single precision data.

if nargin < 2
   number_iterations = [];
end

if nargin < 1
    N = [];
end

if isempty( N )
    N = 100;
end

if isempty( number_iterations )
   number_iterations = 20;
end

% create a full rank, positive definite Hermitian matrix as well as its symmetric
% counterpart.
X = randn( N ) + randn( N )*i;
R = X * X';
S = real( R );
I = zeros( N );

% benchmark each data type and matrix type.
t_double_hermitian = time_inversion_methods( R, I, number_iterations );
t_double_symmetric = time_inversion_methods( S, I, number_iterations );
t_single_hermitian = time_inversion_methods( single( R ), single( I ), number_iterations );
t_single_symmetric = time_inversion_methods( single( S ), single( I ), number_iterations );

% pack things up for the caller.
timings = [t_double_hermitian, t_double_symmetric, t_single_hermitian, t_single_symmetric];

return

function timings = time_inversion_methods( R, I, number_iterations )
% timings = time_inversion_methods( R, I, number_iterations )
%
% Benchmarks the several inversion methods for the specified positive definite
% matrix.  Each method is performed a caller-specified number of times and the
% median of all run-times is returned for the method.  The following methods
% are timed:
%
%   1. Inversion via inv().
%   2. Inversion via heuristic derived factorization and back substitution.
%   3. Inversion via Cholesky factorization and back substitution.
%   4. Inversion via the cholesky_inverse() MEX object.
%
% Takes 3 arguments:
%
%   R                 - Positive definite matrix to invert.
%   I                 - Identity matrix of the same size and class as R.
%   number_iterations - Number of iterations to perform the inversion for
%                       benchmarking purposes.
%
% Returns 1 value:
%
%   timings - Vector, of size 4 x 1, containing the timings, in seconds, of
%             each technique.  The ith entry corresponds to the ith technique
%             listed above.

timings = zeros( 4, 1 );

% individual run timings.  these are condensed down into a scalar value.
local_timings = zeros( number_iterations, 1 );

% naive inverse.  turn off warnings about nearly singular matrices.
W = warning( 'off' );
for iteration_index = 1:number_iterations
    tic
    R_inv = inv( R );
    local_timings(iteration_index) = toc;
end
timings(1) = median( local_timings );
warning( W );

% factorization and back substitution inverse.  turn off warnings about nearly
% singular matrices.
W = warning( 'off' );
for iteration_index = 1:number_iterations
    tic
    R_inv = R \ I;
    local_timings(iteration_index) = toc;
end
timings(2) = median( local_timings );
warning( W );

%  exploiting positive definite structure.
for iteration_index = 1:number_iterations
    tic
    U     = chol( R );
    U_inv = U \ I;
    R_inv = U_inv * U_inv';
    local_timings(iteration_index) = toc;
end
timings(3) = median( local_timings );

% explicit inverse exploiting positive definite structure.
for iteration_index = 1:number_iterations
    tic
    R_inv = cholesky_inverse( R );
    local_timings(iteration_index) = toc;
end
timings(4) = median( local_timings );

return
