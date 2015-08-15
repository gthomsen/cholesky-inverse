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

% logical indicating whether the MEX object is working according to
% specification.  if any test fails, we set this to false.
status = true;

% ensure that our "random" matrices are reproducible to aide in debugging.
%
% XXX: this doesn't actually work.
rand( 'seed', 1 );

% order of the matrices we'll use for this test.
n = 5;

% double and single precision complex and real matrices.
X_double = randn( n ) + randn( n ) * i;
Y_double = real( X_double );

X_single = single( X_double );
Y_single = single( Y_double );

% create full rank, positive definite, Hermitian and symmetric matrices for
% all data types we're going to test.
%
% NOTE: we cast our Hermitian and symmetric to work around the fact that
%       Octave doesn't support matrix operations on integral data.
%
R_single = X_single * X_single';
R_double = X_double * X_double';
S_single = Y_single * Y_single';
S_double = Y_double * Y_double';

S_int8   = int8( S_double );
S_int16  = int16( S_double );
S_int32  = int32( S_double );
S_int64  = int64( S_double );

% begin testing the MEX object.

% test cases for native precision using Hermitian and symmetric matrices.
status = and( status, test_success( @() verify_inversion( R_single ), ...
                                    'Failed to execute with a single precision, Hermitian matrix.' ) );
status = and( status, test_success( @() verify_inversion( R_double ), ...
                                    'Failed to execute with a double precision, Hermitian matrix.' ) );
status = and( status, test_success( @() verify_inversion( S_single ), ...
                                    'Failed to execute with a single precision, symmetric matrix.' ) );
status = and( status, test_success( @() verify_inversion( S_double ), ...
                                    'Failed to execute with a single precision, symmetric matrix.' ) );

% test cases for specified precision using Hermitian matrices.
status = and( status, test_success( @() verify_inversion( R_single, 'single' ), ...
                                    'Failed to execute with a single precision, Hermitian matrix, specifying single precision.' ) );
status = and( status, test_success( @() verify_inversion( R_double, 'single' ), ...
                                    'Failed to execute with a double precision, Hermitian matrix, specifying single precision.' ) );
status = and( status, test_success( @() verify_inversion( R_single, 'double' ), ...
                                    'Failed to execute with a single precision, Hermitian matrix, specifying double precision.' ) );
status = and( status, test_success( @() verify_inversion( R_double, 'double' ), ...
                                    'Failed to execute with a double precision, Hermitian matrix, specifying double precision.' ) );

% test cases for specified precision using symmetric matrices.
status = and( status, test_success( @() verify_inversion( S_single, 'single' ), ...
                                    'Failed to execute with a single precision, symmetric matrix, specifying single precision.' ) );
status = and( status, test_success( @() verify_inversion( S_double, 'single' ), ...
                                    'Failed to execute with a double precision, symmetric matrix, specifying single precision.' ) );
status = and( status, test_success( @() verify_inversion( S_single, 'double' ), ...
                                    'Failed to execute with a single precision, symmetric matrix, specifying double precision.' ) );
status = and( status, test_success( @() verify_inversion( S_double, 'double' ), ...
                                    'Failed to execute with a double precision, symmetric matrix, specifying double precision.' ) );

% calling without parameters is invalid.
status = and( status, test_failure( @() cholesky_inverse, ...
                                    'MEX object allowed no arguments to be supplied.' ) );

% calling with invalid parameters is invalid.  first supply matrices with
% integral data types, then supply invalid precision strings.
for input_matrix = { S_int8, S_int16, S_int32, S_int64 }
    status = and( status, test_failure( @() cholesky_inverse( input_matrix{1} ), ...
                                        'MEX object should not allow inputs that are not single- or double precision.' ) );
end

for type_str = { 'int8', 'int16', 'int32', 'SINGLE', 'DOUBLE', 'FLOAT32', 'FLOAT64' }
    status = and( status, test_failure( @() cholesky_inverse( R_single, type_str{1} ), ...
                                        sprintf( 'MEX object should not allow computation precision ''%s''.', ...
                                                 type_str{1} ) ) );
end

if status
    disp( 'All passed!' )
end

return

function verify_inversion( R, precision )
% verify_inversion( R, precision )
%
% Verifies that the supplied positive definite matrix can be inverted using
% the Cholesky inversion MEX object.  The inverse is compared against the
% inverse computed via back-substitution against the Cholesky factorization,
% as well as how well it computes the identify from the original matrix.
%
% This function makes several calls to assert() to verify that the MEX
% object's inversion is correct.  If a problem occurs during execution, this
% function throws an error and does not return.
%
% Takes 2 arguments:
%
%   R         - Positive definite matrix (Hermitian or symmetric) to invert
%               using Cholesky decomposition.
%   precision - String indicating the precision of the inversion computation.
%               Must be one of the valid strings accepted by
%               cholesky_inverse().
%
% Returns nothing.

% invert using the MEX object.
if nargin < 2
    R_inv_mex = cholesky_inverse( R );
else
    R_inv_mex = cholesky_inverse( R, precision );
end

% invert using the built-in Cholesky decomposition routine and
% back-substitution.
I     = eye( size( R ), class( R ) );
U     = chol( R );
U_inv = U \ I;
R_inv = U_inv * U_inv';

% XXX: the tolerances here are awfully large.  should be set based on the
%      class and the norm or R.
close_enough = norm( (R_inv_mex * R) - I ) < 5e-4;
assert( close_enough );

close_enough = norm( (R_inv_mex - R_inv) ) < 5e-4;
assert( close_enough );

return

function status = test_success( function_handle, error_message )
% status = test_success( function_handle, error_message )
%
% Evaluates the supplied anonymous function and verifies that it successfully
% executes.  If execution fails, displays the supplied error message and returns
% false.  Otherwise, returns true.
%
% Takes 2 arguments:
%
%   function_handle - Function handle, that requires no arguments or output
%                     values, to evaluate for success using feval().
%   error_message   - Error message to display if the function evaluation
%                     fails.
%
% Returns 1 value:
%
%   status - Logical indicating whether the function execution succeeded
%            (true), or failed (false).

status = run_test( function_handle, error_message, true );

return

function status = test_failure( function_handle, error_message )
% status = test_success( function_handle, error_message )
%
% Evaluates the supplied anonymous function and verifies that it fails to
% execute.  If execution succeeds, displays the supplied error message and
% returns false.  Otherwise, returns true.
%
% Takes 2 arguments:
%
%   function_handle - Function handle, that requires no arguments or output
%                     values, to evaluate for failure using feval().
%   error_message   - Error message to display if the function evaluation
%                     succeeds.
%
% Returns 1 value:
%
%   status - Logical indicating whether the function execution failed (true),
%            or succeeded (false).

status = run_test( function_handle, error_message, false );

return

function status = run_test( function_handle, error_message, expected_result )
% status = run_test( function_handle, error_message, expected_result )
%
% Runs a function handle, typically anonymous, and compares the execution
% results against the specified expected result.  Prints the error message to
% standard output and returns failure (false) if the function did not behave
% as expected, or returns success (true) if it did.
%
% Takes 3 arguments:
%
%   function_handle - Function handle, that requires no arguments or output values,
%                     to evaluate using feval().
%   error_message   - Error message to display if the function evaluation does not
%                     behave according to expected_result.
%   expected_result - Optional logical specifying whether the test should pass
%                     (true) or fail (false).  If omitted, defaults to true.
%
% Returns 1 value:
%
%   status - Logical indicating whether the test behaved according to
%            expected_result.  True if the test executed according to
%            expected_result, false otherwise.

if nargin < 3
    expected_result = [];
end

if isempty( expected_result )
    expected_result = true;
end

% we build a failure message from an error stack trace under Octave.  identify
% the "bottom" of the error stack that we wish to print out.
if expected_result == true
    stack_index = 3;
else
    % XXX: this is likely wrong as it hasn't been thoroughly tested.
    stack_index = 3;
end

% we start out assuming this test has passed.
status = true;

if expected_result == true
    % we expect our try/catch block to execute without problems.  set the
    % status to failure if we get into the catch, and report where we went
    % awry.
    if is_octave
        try
            feval( function_handle );
        catch
            disp( error_message );
            err = lasterror;
            disp( sprintf( 'Test failed - %s\n%s:%d -> %s:%d\n', ...
                           err.message, ...
                           err.stack(stack_index).file, err.stack(stack_index).line, ...
                           err.stack(stack_index-2).file, err.stack(stack_index-2).line ) );
            status = false;
        end
    else
        try
            feval( function_handle );
        catch me
            disp( error_message );
            getReport( me )
            status = false;
        end
    end
else
    % we expect our try/catch block to fail.  set the status to failure if we
    % get past the feval(), and report where we went awry.
    if is_octave
        try
            feval( function_handle );
            status = false;
            disp( sprintf( 'Test executed when it should have failed - %s\n%s:%d -> %s:%d\n', ...
                           err.message, ...
                           err.stack(4).file, err.stack(4).line, ...
                           err.stack(2).file, err.stack(2).line ) );
        catch
        end
    else
        try
            feval( function_handle );
            disp( error_message );
            getReport( me )
            status = false;
        catch me
        end
    end
end

return

function flag = is_octave()
% flag = is_octave()
%
% Predicate indicating whether the interpreter is Octave.  Returns true if so,
% false if running within MATLAB.
%
% Takes no arguments.
%
% Returns 1 value:
%
%   flag - Logical indicating whether execution occurs within Octave.  True if
%          so.  False otherwise.

% keep the result of our check in memory since this function won't change it's
% return value once it's been determined.
persistent existence_result

if isempty( existence_result )
   existence_result = exist( 'OCTAVE_VERSION', 'builtin' );
end

% we're running under Octave if OCTAVE_VERSION is a builtin.
flag = existence_result == 5;

return
