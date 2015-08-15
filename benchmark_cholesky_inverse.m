function timings = benchmark_cholesky_inverse( N, number_iterations )
% timings = benchmark_cholesky_inverse( N, number_iterations )
%
% Benchmarks the Cholesky inverse MEX object against other alternatives for
% computing the explicit inverse within Matlab.
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
%   timings -
%

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

X = randn( N ) + randn( N )*i;
I = zeros( N );
R = X * X';

t_double = time_inversion_methods( R, I, number_iterations );
t_single = time_inversion_methods( single( R ), single( I ), number_iterations );

timings = [t_double, t_single];

return

function timings = time_inversion_methods( R, I, number_iterations )

% timings hold times for:
%
%   1. naive inverse
%   2. hereustic derived factorization + back substitution
%   3. Cholesky factorization + back substitution
%   4. Cholesky inverse using MEX
%
timings = zeros( 4, 1 );

% individual run timings.  these are condensed down into a scalar value.
local_timings = zeros( number_iterations, 1 );

% naive inverse.
W = warning( 'off' );
for iteration_index = 1:number_iterations
    tic
    R_inv = inv( R );
    local_timings(iteration_index) = toc;
end
timings(1) = median( local_timings );
warning( W );

% factorization and back substitution inverse.
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
