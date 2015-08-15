Cholesky Inverse MEX Object
===========================

Provides a MEX object that efficiently inverts a positive definite matrix
using Cholesky factorization, with optional control over the precision by
which the inversion is performed.  This provides a single step inversion in
MATLAB and Octave that is faster than the constituent parts within the
interpreter.  Single and double precision, Hermitian and symmetric matrices
are supported.


Documentation
-------------
The MEX object, the benchmark, and regression test functions all have help
describing their behavior, inputs, and outputs.

    >> R_inv = cholesky_inverse( R );
    >> R_inv = cholesky_inverse( R, 'double' );

Verification that the MEX object is function properly can be done by running
the regression test:

    >> status = test_cholesky_inverse()

Timing information for the MEX object can be measured by running the benchmark:

    >> timings = benchmark_cholesky_inverse( 100 )

The size of the matrix inverted can be specified by changing `100` to the
desired value.

Installation
------------
The MEX object may be compiled from the command line or within MATLAB/Octave.
Below are instructions for compiling within each of the interpreters.

=== MATLAB ===

    % optimized
    >> mex -largeArrayDims -lmwlapack -O cholesky_inverse.c
    
    % debug
    >> mex -largeArrayDims -lmwlapack -g cholesky_inverse.c
    
    % profiling
    >> mex -largeArrayDims -lmwlapack -O -g cholesky_inverse.c

=== Octave ===

    % optimized
    >> [output, status] = mkoctfile( '--mex', 'cholesky_inverse.c' );
    
    % debug
    >> [output, status] = mkoctfile( '--mex', '-g', 'cholesky_inverse.c' );

Licensing
---------
