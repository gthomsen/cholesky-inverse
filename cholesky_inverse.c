#include <string.h>
#include <stdio.h>

#include "mex.h"

/*
  MEX object that allows inversion of a positive definite matrix (either
  symmetric or Hermitian) via LAPACK's potrf()/potri() routines.  This routine
  creates a new matrix, interleaves the real and imaginary portions of the
  supplied matrix, inverts it efficiently, and then copies the result into a
  newly created matrix.  Care is taken to minimize buffer copies so this is
  faster the equivalent operations in MATLAB (Cholesky decomposition via
  chol(), a linear solve for the factor's inverse, and explicit formation of
  the full inverse).

  Initial benchmarks indicate this object is roughly twice as fast as a simple
  linear solve against the identity matrix, and ~30% faster a Cholesky
  factorization, factor inversion, and full matrix inverse calculation.
 */


#ifndef HAVE_OCTAVE
/* pull in mxCreateUninitNumericArray() if we're building a Matlab MEX object.
   Octave does not implement this part of the MEX API in v3.6.4.

   NOTE: we need to include this after mex.h so we can see the HAVE_OCTAVE
         pre-processor symbol. */
#include "matrix.h"
#endif

/* indices for the input arguments and output value the MEX object accepts. */
#define INPUT_X_INDEX         0
#define INPUT_PRECISION_INDEX 1
#define OUTPUT_X_INV_INDEX    0

/* valid precision strings for the optional, second argument the MEX object
   accepts. */
#define PRECISION_DOUBLE_STR  "double"
#define PRECISION_FLOAT32_STR "float32"
#define PRECISION_FLOAT64_STR "float64"
#define PRECISION_SINGLE_STR  "single"

/* TODO:
   1. Hold onto the internal inversion buffer.
   2. Only copy the upper triangular portion of the matrix since LAPACK
      doesn't touch the lower portion.
   3. Benchmark against packed representations.  Copy only what we need.
   4. Return second parameter indicating the sub-matrix that isn't positive
      definite.
 */

/* provide declarations for the LAPACK routines needed to invert a positive
   definite matrix using Cholesky decomposition.

   Octave does not provide a compatibility layer and requires mapping the
   unadorned LAPACK names to the underlying Fortran symbols that have
   trailing "_" on the names.

      NOTE: This layer assumes that we're compiled with mkoctfile so that
            the integers provided to LAPACK match what is expected.

   MATLAB does provide a compatibility layer which handles mapping to
   the underlying LAPACK.  symbol names and integer sizes are properly
   handled behind the scenes. */
#ifdef HAVE_OCTAVE
/* Cholesky decomposition. */
extern void spotrf_( char *uplo, int *n, float* a, int *lda, int *info );
extern void dpotrf_( char *uplo, int *n, double* a, int *lda, int *info );
extern void cpotrf_( char *uplo, int *n, float* a, int *lda, int *info );
extern void zpotrf_( char *uplo, int *n, double* a, int *lda, int *info );

/* Inversion using a Cholesky factorization. */
extern void spotri_( char *uplo, int *n, float* a, int *lda, int *info );
extern void dpotri_( char *uplo, int *n, double* a, int *lda, int *info );
extern void cpotri_( char *uplo, int *n, float* a, int *lda, int *info );
extern void zpotri_( char *uplo, int *n, double* a, int *lda, int *info );

#define spotrf spotrf_
#define dpotrf dpotrf_
#define cpotrf cpotrf_
#define zpotrf zpotrf_

#define spotri spotri_
#define dpotri dpotri_
#define cpotri cpotri_
#define zpotri zpotri_

#else  /* MATLAB */
/* this pulls in wrappers for the Fortran implementations without the "_"
   suffix.  note that all integers for this interface are ptrdiff_t which
   match the pointer size during compilation. */
#include "lapack.h"
#endif

void copy_split_to_interleaved( void * __restrict__ interleaved_buffer,
                                const void * __restrict__ real_buffer,
                                const void * __restrict__ imaginary_buffer,
                                int number_elements,
                                mxClassID source_type,
                                mxClassID destination_type )
{
    /* linear indices to interleave the real/imaginary portions of the matrix
       into a single buffer suitable for use by LAPACK. */
    int element_index     = 0;
    int interleaved_index = 0;

    /* are we simply splitting one buffer into two? */
    if( source_type == destination_type )
    {
        /* are we copying complex values?  if so, we need to de-interleave
           them. */
        if( imaginary_buffer != NULL )
        {
            if( source_type == mxDOUBLE_CLASS )
            {
                /* doubles to doubles. */
                double * __restrict__ real        = (double *)real_buffer;
                double * __restrict__ imaginary   = (double *)imaginary_buffer;
                double * __restrict__ interleaved = (double *)interleaved_buffer;

                for( element_index = 0;
                     element_index < number_elements;
                     element_index++, interleaved_index += 2 )
                {
                    interleaved[interleaved_index]     = real[element_index];
                    interleaved[interleaved_index + 1] = imaginary[element_index];
                }
            }
            else
            {
                /* floats to floats. */
                float * __restrict__ real        = (float *)real_buffer;
                float * __restrict__ imaginary   = (float *)imaginary_buffer;
                float * __restrict__ interleaved = (float *)interleaved_buffer;

                for( element_index = 0;
                     element_index < number_elements;
                     element_index++, interleaved_index += 2 )
                {
                    interleaved[interleaved_index]     = real[element_index];
                    interleaved[interleaved_index + 1] = imaginary[element_index];
                }
            }
        }
        else
        {
            /* our source and destination types are the same, and we aren't
               de-interleaving things.  simply memcpy() the data. */
            memcpy( interleaved_buffer, real_buffer,
                    number_elements * (source_type == mxDOUBLE_CLASS ?
                                       sizeof( double ) : sizeof( float )) );
        }
    }
    /* or splitting into two buffers that are a different type than the
       source? */
    else
    {
        /* are we copying complex values?  if so, we need to de-interleave
           them. */
        if( imaginary_buffer != NULL )
        {
            if( source_type == mxDOUBLE_CLASS )
            {
                /* complex doubles to complex float. */
                double * __restrict__ real        = (double *)real_buffer;
                double * __restrict__ imaginary   = (double *)imaginary_buffer;
                float * __restrict__  interleaved = (float *)interleaved_buffer;

                for( element_index = 0;
                     element_index < number_elements;
                     element_index++, interleaved_index += 2 )
                {
                    interleaved[interleaved_index]     = real[element_index];
                    interleaved[interleaved_index + 1] = imaginary[element_index];
                }
            }
            else
            {
                /* complex float to complex double */
                float * __restrict__  real        = (float *)real_buffer;
                float * __restrict__  imaginary   = (float *)imaginary_buffer;
                double * __restrict__ interleaved = (double *)interleaved_buffer;

                for( element_index = 0;
                     element_index < number_elements;
                     element_index++, interleaved_index += 2 )
                {
                    interleaved[interleaved_index]     = real[element_index];
                    interleaved[interleaved_index + 1] = imaginary[element_index];
                }
            }
        }
        else
        {
            /* our source and destination types are different, but we're not
               de-interleaving things.  copy and convert the data. */
            if( source_type == mxDOUBLE_CLASS )
            {
                double * __restrict__ real        = (double *)real_buffer;
                float * __restrict__  interleaved = (float *)interleaved_buffer;

                for( element_index = 0; element_index < number_elements; element_index++ )
                    interleaved[element_index] = real[element_index];
            }
            else
            {
                float * __restrict__  real        = (float *)real_buffer;
                double * __restrict__ interleaved = (double *)interleaved_buffer;

                for( element_index = 0; element_index < number_elements; element_index++ )
                    interleaved[element_index] = real[element_index];
            }
        }
    }
}

void copy_interleaved_to_split( void * __restrict__ real_buffer,
                                void * __restrict__ imaginary_buffer,
                                const void * __restrict__ interleaved_buffer,
                                int n,
                                mxClassID source_type,
                                mxClassID destination_type )
{
    /* indices used to index the matrices linearly both in column-
       (element_index) and row-major (mirror_index) order so that we can
       operate on both the upper and lower triangular portions of the matrix
       without complex arithmetic. */
    int element_index = 0;
    int mirror_index  = 0;

    /* linear index into the column-major real/imaginary buffers from the
       input matrix. */
    /* XXX: rename this */
    int real_index    = 0;

    /* row/column indices used to iterate through the matrix we're copying
       and (conjugate) transposing. */
    int row_index    = 0;
    int column_index = 0;

    /* are we simply splitting one buffer into to? */
    if( source_type == destination_type )
    {
        if( imaginary_buffer != NULL )
        {
            if( source_type == mxDOUBLE_CLASS )
            {
                /* complex doubles to complex doubles. */
                double * __restrict__ real        = (double *)real_buffer;
                double * __restrict__ imaginary   = (double *)imaginary_buffer;
                double * __restrict__ interleaved = (double *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    mirror_index = column_index * 2;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index += 2, mirror_index += n * 2, real_index++ )
                    {
                        real[real_index]      = interleaved[element_index];
                        imaginary[real_index] = interleaved[element_index + 1];
                    }

                    /* lower triangle -> conjugate transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index += 2, mirror_index += n * 2, real_index++ )
                    {
                        real[real_index]      =  interleaved[mirror_index];
                        imaginary[real_index] = -interleaved[mirror_index + 1];
                    }
                }
            }
            else
            {
                /* complex floats to complex floats. */
                float * __restrict__ real        = (float *)real_buffer;
                float * __restrict__ imaginary   = (float *)imaginary_buffer;
                float * __restrict__ interleaved = (float *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    mirror_index = column_index * 2;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index += 2, mirror_index += n * 2, real_index++ )
                    {
                        real[real_index]      = interleaved[element_index];
                        imaginary[real_index] = interleaved[element_index + 1];
                    }

                    /* lower triangle -> conjugate transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index += 2, mirror_index += n * 2, real_index++ )
                    {
                        real[real_index]      =  interleaved[mirror_index];
                        imaginary[real_index] = -interleaved[mirror_index + 1];
                    }
                }
            }
        }
        else
        {
            if( source_type == mxDOUBLE_CLASS )
            {
                /* real doubles to real doubles. */
                double * __restrict__ real        = (double *)real_buffer;
                double * __restrict__ interleaved = (double *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    mirror_index = column_index;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[element_index];

                    /* lower triangle -> transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[mirror_index];
                }
            }
            else
            {
                /* real floats to floats. */
                float * __restrict__ real        = (float *)real_buffer;
                float * __restrict__ interleaved = (float *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    mirror_index = column_index;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[element_index];

                    /* lower triangle -> transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[mirror_index];
                }
            }
        }
    }
    /* or splitting into two buffers that are a different type than the
       source? */
    else
    {
        if( imaginary_buffer != NULL )
        {
            if( source_type == mxDOUBLE_CLASS )
            {
                /* complex doubles to complex floats. */
                float * __restrict__  real        = (float *)real_buffer;
                float * __restrict__  imaginary   = (float *)imaginary_buffer;
                double * __restrict__ interleaved = (double *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    mirror_index = column_index * 2;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index += 2, mirror_index += 2 * n, real_index++ )
                    {
                        real[real_index]      = interleaved[element_index];
                        imaginary[real_index] = interleaved[element_index + 1];
                    }

                    /* lower triangle -> conjugate transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index += 2, mirror_index += 2 * n, real_index++ )
                    {
                        real[real_index]      =  interleaved[mirror_index];
                        imaginary[real_index] = -interleaved[mirror_index + 1];
                    }
                }
            }
            else
            {
                /* complex floats to complex doubles. */
                double * __restrict__ real        = (double *)real_buffer;
                double * __restrict__ imaginary   = (double *)imaginary_buffer;
                float * __restrict__  interleaved = (float *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    mirror_index = column_index * 2;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index += 2, mirror_index += n * 2, real_index++ )
                    {
                        real[real_index]      = interleaved[element_index];
                        imaginary[real_index] = interleaved[element_index + 1];
                    }

                    /* lower triangle -> conjugate transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index += 2, mirror_index += n * 2, real_index++ )
                    {
                        real[real_index]      =  interleaved[mirror_index];
                        imaginary[real_index] = -interleaved[mirror_index + 1];
                    }
                }
            }
        }
        else
        {
            if( source_type == mxDOUBLE_CLASS )
            {
                /* real doubles to real floats. */
                float * __restrict__  real        = (float *)real_buffer;
                double * __restrict__ interleaved = (double *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    mirror_index = column_index;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[element_index];

                    /* lower triangle -> transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[mirror_index];
                }
            }
            else
            {
                /* real floats to real doubles. */
                double * __restrict__ real        = (double *)real_buffer;
                float * __restrict__  interleaved = (float *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    mirror_index = column_index;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[element_index];

                    /* lower triangle -> transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[mirror_index];
                }
            }
        }
    }
}

void invert_matrix( void *matrix_buffer, int N, mxClassID computation_class, int complexity_flag )
{
    /* dimension of the matrix and status variable for the LAPACK calls. */
#ifdef HAVE_OCTAVE
    int     n               = N;
    int     lapack_status   = 0;
#else  /* MATLAB */
    ptrdiff_t n             = N;
    ptrdiff_t lapack_status = 0;
#endif

    /* our factorization operates on the upper triangular portion of the
       matrix. */
    char uplo = 'U';

    if( computation_class == mxUNKNOWN_CLASS )
        return;

    /* do the Cholesky factorization based on the data type and complexity.
       throw an error indicating that the matrix being inverted isn't positive
       definite if LAPACK concludes that. */
    if( computation_class == mxDOUBLE_CLASS )
    {
        /* factor */
        if( complexity_flag )
            zpotrf( &uplo, &n, matrix_buffer, &n, &lapack_status );
        else
            dpotrf( &uplo, &n, matrix_buffer, &n, &lapack_status );

        if( lapack_status != 0 )
        {
            mxFree( matrix_buffer );
            mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:lapack",
                               "Failed to factorize X (%d).",
                               lapack_status );
        }

        /* invert */
        if( complexity_flag )
            zpotri( &uplo, &n, matrix_buffer, &n, &lapack_status );
        else
            dpotri( &uplo, &n, matrix_buffer, &n, &lapack_status );

        if( lapack_status != 0 )
        {
            mxFree( matrix_buffer );
            mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:lapack",
                               "Failed to invert X (%d).",
                               lapack_status );
        }
    }
    else
    {
        if( complexity_flag )
            cpotrf( &uplo, &n, matrix_buffer, &n, &lapack_status );
        else
            spotrf( &uplo, &n, matrix_buffer, &n, &lapack_status );

        if( lapack_status != 0 )
        {
            mxFree( matrix_buffer );
            mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:lapack",
                               "Failed to factorize X (%d).",
                               lapack_status );
        }

        /* invert */
        if( complexity_flag )
            cpotri( &uplo, &n, matrix_buffer, &n, &lapack_status );
        else
            spotri( &uplo, &n, matrix_buffer, &n, &lapack_status );

        if( lapack_status != 0 )
        {
            mxFree( matrix_buffer );
            mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:lapack",
                               "Failed to invert X (%d).",
                               lapack_status );
        }
    }

    return;
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    int       n            = 0;
    int       complexity_flag = 0;

    mxClassID computation_class = mxUNKNOWN_CLASS;

    size_t matrix_size = 0;

    /* holds the dimensions of the matrix being inverted. */
    const mwSize *input_dimensions = NULL;

    /* dimensions of the inverse we return.  the MEX interface has changed and
       Octave does not yet support the newest version (2015a) as of
       2015/08/14. */
#ifdef HAVE_OCTAVE
    mwSize output_dimensions[2];
#else
    size_t output_dimensions[2];
#endif

    /* pointer to the inverted matrix returned by invert_matrix(). */
    void *inverted_matrix = NULL;

    /* Check for proper number of input and output arguments */
    if( nrhs < 1 || nrhs > 2 )
        mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:invalidNumInputs",
                           "Either one or two input arguments required." );

    if( nlhs > 1 )
        mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:maxlhs",
                           "Too many output arguments.  No more than one may be requested." );

    /* get the dimension of X and verify that X is square. */
    n = mxGetM( prhs[INPUT_X_INDEX] );
    if( n != mxGetN( prhs[INPUT_X_INDEX] ) )
        mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:invalidInput",
                           "X must be square." );

    /* get the class and complexity of X */
    complexity_flag = mxIsComplex( prhs[INPUT_X_INDEX] );

    /* verify that X is either single or double precision so we can simplify
       our lives. */
    switch( mxGetClassID( prhs[INPUT_X_INDEX] ) )
    {
    case mxDOUBLE_CLASS:
    case mxSINGLE_CLASS:
        break;
    default:
        mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:invalidInput",
                           "X must be either single or double precision." );
    }

    /* get the precision requested for the output, default to the class of X */
    if( nrhs == 2 )
    {
        char precision_string[128];

        if( 0 != mxGetString( prhs[INPUT_PRECISION_INDEX],
                              precision_string,
                              sizeof( precision_string ) ) )
            mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:invalidPrecisionType",
                               "Failed to acquire the precision argument as a string." );

        if( 0 == strcmp( precision_string, PRECISION_DOUBLE_STR ) ||
            0 == strcmp( precision_string, PRECISION_FLOAT64_STR ) )
            computation_class = mxDOUBLE_CLASS;
        else if( 0 == strcmp( precision_string, PRECISION_SINGLE_STR ) ||
                 0 == strcmp( precision_string, PRECISION_FLOAT32_STR ) )
            computation_class = mxSINGLE_CLASS;
        else
            mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:invalidPrecisionType",
                               "Precision string must be one of '" PRECISION_DOUBLE_STR "', '"
                                PRECISION_FLOAT64_STR "', '" PRECISION_SINGLE_STR "', '"
                                PRECISION_FLOAT32_STR "'." );
    }
    else
        /* the user didn't specify a precision to do the computations in, so
           do them in the same precision the data are. */
        computation_class = mxGetClassID( prhs[INPUT_X_INDEX] );

    /* allocate a buffer for the inversion. */
    matrix_size = (n * n *
                   (complexity_flag ? 2 : 1) *
                   (computation_class == mxDOUBLE_CLASS ? sizeof( double ) : sizeof( float )));
    if( NULL == (inverted_matrix = mxMalloc( matrix_size )) )
        mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:memoryAllocation",
                           "Failed to allocate %lu bytes for the inversion buffer.",
                           matrix_size );

    /* create an interleaved buffer of the appropriate data type allows us to
       use it with LAPACK.  this properly handles input matrices that are
       either real or complex. */
    copy_split_to_interleaved( inverted_matrix,
                               mxGetPr( prhs[INPUT_X_INDEX] ),
                               mxGetPi( prhs[INPUT_X_INDEX] ),
                               n * n,
                               mxGetClassID( prhs[INPUT_X_INDEX] ),
                               computation_class );

    /* invert the matrix. */
    invert_matrix( inverted_matrix, n, computation_class, complexity_flag );

    /* if the user didn't want the output, return now. */
    if( nlhs < 1 )
    {
        mxFree( inverted_matrix );
        return;
    }

    /* allocate the output variable with the user specified precision and X's
       complexity.  since we'll immediately copy our interleaved matrix into
       this variable's buffer(s), we request uninitialized data when
       possible. */
    input_dimensions     = mxGetDimensions( prhs[INPUT_X_INDEX] );
    output_dimensions[0] = input_dimensions[0];
    output_dimensions[1] = input_dimensions[1];

#ifdef HAVE_OCTAVE
    if( NULL == (plhs[OUTPUT_X_INV_INDEX] = mxCreateNumericArray(       2,
#else
    if( NULL == (plhs[OUTPUT_X_INV_INDEX] = mxCreateUninitNumericArray( 2,
#endif
                                                                        output_dimensions,
                                                                        mxGetClassID( prhs[INPUT_X_INDEX] ),
                                                                        (complexity_flag ? mxCOMPLEX : mxREAL) )) )
    {
        mxFree( inverted_matrix );
        mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:memoryAllocation",
                           "Failed to allocate the output inverse matrix." );
    }

    /* copy the data from contiguous to interleaved.  this properly handles
       mirroring the data from upper triangular to full matrix as well. */
    copy_interleaved_to_split( mxGetPr( plhs[OUTPUT_X_INV_INDEX] ),
                               mxGetPi( plhs[OUTPUT_X_INV_INDEX] ),
                               inverted_matrix,
                               mxGetM( plhs[OUTPUT_X_INV_INDEX] ),
                               computation_class,
                               mxGetClassID( plhs[OUTPUT_X_INV_INDEX] ) );

    mxFree( inverted_matrix );

    return;
}
