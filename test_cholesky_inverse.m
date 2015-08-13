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

% calling without output arguments is valid.
try
    R_single
    R_inv_single      = cholesky_inverse( R_single );

    U                 = chol( R_single );
    U_inv             = U \ eye( size( R_single ) );
    R_inv_single_real = U_inv*U_inv';

    % XXX: this tolerance is awfully large.
    close_enough = (abs( R_inv_single_real * R_single ) - single( eye( size( R_single ) ) )) < 1e-5;
    assert( all( close_enough(:) ) );
    close_enough = (R_inv_single_real - R_inv_single) < 1e-5;
    assert( all( close_enough(:) ) );
%    assert( abs( R_inv_single_real * R_single ), single( eye( size( R_single ) ) ), -5e-6 );
%    assert( R_inv_single_real, R_inv_single, -1e-6 );
catch me
    status = false;
    getReport( me )
%    msg = lasterror.message;
%    disp( msg);
    keyboard
    error( 'cholesky_inverse() failed to execute with a single precision, Hermitian matrix.' );
end

keyboard

try
    cholesky_inverse( R_double );
catch
    status = false;
    error( 'cholesky_inverse() failed to execute with a double precision, Hermitian matrix.' );
end

try
    cholesky_inverse( S_single );
catch
    status = false;
    error( 'cholesky_inverse() failed to execute with a single precision, symmetric matrix.' );
end

try
    cholesky_inverse( S_double );
catch
    status = false;
    error( 'cholesky_inverse() failed to execute with a double precision, symmetrix matrix.' );
end

% calling with invalid parameters is invalid.
for type_str = { 'int8', 'int16', 'int32', 'SINGLE', 'DOUBLE', 'FLOAT32', 'FLOAT64' }
  try
      cholesky_inverse( R_single, type_str{1} );

      status = false;
  catch
  end
end

R_inv_double = cholesky_inverse( R_double, 'double' );
R_inv_single = cholesky_inverse( R_single, 'single' )
R_inv_single = cholesky_inverse( R_single, 'double' );

return
