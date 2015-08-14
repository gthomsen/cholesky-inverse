function status = test_cholesky_inverse()
% status = test_cholesky_inverse()
%
% Tests the cholesky_inverse() MEX object to verify that it operates
% correctly.  The interface is verified to work as advertised and the computed
% inverse is verified to be identical to the results to within precision XXX.
%
% Takes no arguments.
%
% Returns 1 value:
%
%   status - Logical indicating whether the cholesky_inverse() object works
%            as advertised.  False if its behavior differs from the expected,
%            true otherwise.
%

status = true;

rand( 'seed', 1 );

n = 5;

X_double = randn( n ) + randn( n ) * i;
Y_double = real( X_double );

Y_int8   = int8( Y_double );
Y_int16  = int16( Y_double );
Y_int32  = int32( Y_double );
Y_int64  = int64( Y_double );

X_single = single( X_double );
Y_single = single( Y_double );

% NOTE: we compute a positive definite, Hermitian matrix in single and double
%       precision and then cast it to integral data types.  this works around
%       the fact that Octave doesn't support matrix operations on integral
%       data.
R_single = X_single * X_single';
R_double = X_double * X_double';
S_single = Y_single * Y_single';
S_double = Y_double * Y_double';

S_int8   = int8( S_double );
S_int16  = int16( S_double );
S_int32  = int32( S_double );

% calling without parameters is invalid.
try
    cholesky_inverse;

    status = false;
    error( 'cholesky_inverse() executed without any parameters.' );
catch
end

% test cases for native precision using Hermitian and symmetric matrices.
run_test( @() verify_inversion( R_single ), ...
        'Failed to execute with a single precision, Hermitian matrix.' );
run_test( @() verify_inversion( R_double ), ...
        'Failed to execute with a double precision, Hermitian matrix.' );
run_test( @() verify_inversion( S_single ), ...
        'Failed to execute with a single precision, symmetric matrix.' );
run_test( @() verify_inversion( S_double ), ...
        'Failed to execute with a single precision, symmetric matrix.' );

% test cases for specified precision using Hermitian matrices.
run_test( @() verify_inversion( R_single, 'single' ), ...
        'Failed to execute with a single precision, Hermitian matrix.' );
run_test( @() verify_inversion( R_double, 'single' ), ...
        'Failed to execute with a double precision, Hermitian matrix.' );
run_test( @() verify_inversion( R_single, 'double' ), ...
        'Failed to execute with a single precision, Hermitian matrix.' );
run_test( @() verify_inversion( R_double, 'double' ), ...
        'Failed to execute with a double precision, Hermitian matrix.' );

% test cases for specified precision using symmetric matrices.
run_test( @() verify_inversion( S_single, 'single' ), ...
        'Failed to execute with a single precision, symmetric matrix.' );
run_test( @() verify_inversion( S_double, 'single' ), ...
        'Failed to execute with a single precision, symmetric matrix.' );
run_test( @() verify_inversion( S_single, 'double' ), ...
        'Failed to execute with a single precision, symmetric matrix.' );
run_test( @() verify_inversion( S_double, 'double' ), ...
        'Failed to execute with a single precision, symmetric matrix.' );

keyboard


% calling with invalid parameters is invalid.
for type_str = { 'int8', 'int16', 'int32', 'SINGLE', 'DOUBLE', 'FLOAT32', 'FLOAT64' }
  try
      cholesky_inverse( R_single, type_str{1} );

      status = false;
  catch
  end
end

return

function verify_inversion( R )
% XXX: swap the names for R_inv_real and R_inv
I = eye( size( R ), class( R ) );

% invert using the MEX object.
R_inv = cholesky_inverse( R );

% invert using the built-in Cholesky decomposition routine and
% back-substitution.
U          = chol( R );
U_inv      = U \ I;
R_inv_real = U_inv * U_inv';

% XXX: this tolerance is awfully large.  should be set based on the class.
close_enough = (abs( R_inv_real * R ) - I)  < 1e-5;
assert( all( close_enough(:) ) );

R_inv_real - R_inv;
close_enough = (R_inv_real - R_inv) < 1e-5;
assert( all( close_enough(:) ) );

%    assert( abs( R_inv_single_real * R_single ), single( eye( size( R_single ) ) ), -5e-6 );
%    assert( R_inv_single_real, R_inv_single, -1e-6 );

return

function run_test( function_handle )

if is_octave
    try
        feval( function_handle );
    catch
        err = lasterror;
        disp( sprintf( 'Test failed - %s\n%s:%d -> %s:%d\n', ...
                       err.message, ...
                       err.stack(4).file, err.stack(4).line, ...
                       err.stack(2).file, err.stack(2).line ) );
    end
else
    try
        feval( function_handle );
    catch me
        getReport( me );
        disp( 'Failed to evaluate the function' );
    end
end

return

function flag = is_octave()

persistent existence_result

if isempty( existence_result )
   existence_result = exist( 'OCTAVE_VERSION', 'builtin' );
end

flag = existence_result ~= 0;

return
