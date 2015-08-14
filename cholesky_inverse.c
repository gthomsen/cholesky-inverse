#include <string.h>
#include <stdio.h>

#include "mex.h"

#ifndef HAVE_OCTAVE
/* pull in mxCreateUninitNumericArray() if we're building a Matlab MEX object.
   Octave does not implement this part of the MEX API in v3.6.4.

   NOTE: we need to include this after mex.h so we can see the HAVE_OCTAVE
         pre-processor symbol. */
#include "matrix.h"
#endif

/* mexAtExit( func ) - void func( void ) */

#define INPUT_X_INDEX         0
#define INPUT_PRECISION_INDEX 1
#define OUTPUT_X_INV_INDEX    0

#define PRECISION_DOUBLE_STR  "double"
#define PRECISION_FLOAT32_STR "float32"
#define PRECISION_FLOAT64_STR "float64"
#define PRECISION_SINGLE_STR  "single"

#define PRECISION_DOUBLE_ID   0
#define PRECISION_SINGLE_ID   1

/* TODO:
   1. Hold onto the internal inversion buffer.
   2. Only copy the upper triangular portion of the matrix since LAPACK
      doesn't touch the lower portion.
   3. Benchmark against packed representations.  Copy only what we need.
   4. Return second parameter indicating the sub-matrix that isn't positive
      definite.
 */

#ifdef HAVE_OCTAVE
extern void spotrf_( char *uplo, int *n, float* a, int *lda, int *info );
extern void dpotrf_( char *uplo, int *n, double* a, int *lda, int *info );
extern void cpotrf_( char *uplo, int *n, float* a, int *lda, int *info );
extern void zpotrf_( char *uplo, int *n, double* a, int *lda, int *info );

extern void spotri_( char *uplo, int *n, float* a, int *lda, int *info );
extern void dpotri_( char *uplo, int *n, double* a, int *lda, int *info );
extern void cpotri_( char *uplo, int *n, float* a, int *lda, int *info );
extern void zpotri_( char *uplo, int *n, double* a, int *lda, int *info );

/* XXX: remove these */
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

void print_square_matrix_float( float *A, int n, int is_complex, char uplo )
{
    int element_index = 0;
    int row_index     = 0;
    int column_index  = 0;

    if( is_complex )
    {
        for( row_index = 0; row_index < n; row_index++ )
        {
            for( column_index = 0; column_index < n; column_index++ )
            {
                /* convert from row-major to column-major indexing. */
                element_index = ((column_index * n) + row_index) * 2;

                /* are we printing an upper triangular matrix? */
                if( uplo == 'U' || uplo == 'u' )
                {
                    if( column_index >= row_index )
                        printf( "   %+8.5f + %+8.5fi",
                                A[element_index],
                                A[element_index + 1] );
                    else
                        printf( "   %+8.5f + %+8.5fi",
                                0., 0. );
                }
                /* a lower triangular matrix? */
                else if( uplo == 'L' || uplo == 'l' )
                {
                    if( column_index <= row_index )
                        printf( "   %+8.5f + %+8.5fi",
                                A[element_index],
                                A[element_index + 1] );
                    else
                        printf( "   %+8.5f + %+8.5fi",
                                0., 0. );
                }
                /* or the full matrix? */
                else
                    printf( "   %+8.5f + %+8.5fi",
                            A[element_index],
                            A[element_index + 1] );
            }
            printf( "\n" );
        }
    }
    else
    {
    }
}

void copy_double_to_float( float * __restrict__ destination, const double * __restrict__ source,
                           int number_elements )
{
    int element_index = 0;

    for( element_index = 0; element_index < number_elements; element_index++ )
        destination[element_index] = source[element_index];
}

void copy_float_to_double( double * __restrict__ destination, const float * __restrict__ source,
                           int number_elements )
{
    int element_index = 0;

    for( element_index = 0; element_index < number_elements; element_index++ )
        destination[element_index] = source[element_index];
}

void copy_split_to_interleaved( void * __restrict__ interleaved_buffer,
                                const void * __restrict__ real_buffer,
                                const void * __restrict__ imaginary_buffer,
                                int number_elements,
                                mxClassID source_type,
                                mxClassID destination_type )
{
    int element_index   = 0;
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
                copy_double_to_float( interleaved_buffer, real_buffer, number_elements );
            else
                copy_float_to_double( interleaved_buffer, real_buffer, number_elements );
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
    int element_index   = 0;
    int real_index      = 0;

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
                    int mirror_index = column_index * 2;

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
                    int mirror_index = column_index * 2;

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
                    int mirror_index = column_index;

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
                    int mirror_index = column_index;

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
                    int mirror_index = column_index * 2;

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
                    int mirror_index = column_index * 2;

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
                    int mirror_index = column_index;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[element_index];

                    /* lower triangle -> conjugate transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] =  interleaved[mirror_index];
                }
            }
            else
            {
                /* real floats to real doubles. */
                double * __restrict__ real        = (double *)real_buffer;
                float * __restrict__  interleaved = (float *)interleaved_buffer;

                for( column_index = 0; column_index < n; column_index++ )
                {
                    int mirror_index = column_index;

                    /* upper triangle -> copy */
                    for( row_index = 0;
                         row_index < column_index;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] = interleaved[element_index];

                    /* lower triangle -> conjugate transpose */
                    for( ;
                         row_index < n;
                         row_index++, element_index++, mirror_index += n, real_index++ )
                        real[real_index] =  interleaved[mirror_index];
                }
            }
        }
    }
}

void *invert_matrix( const mxArray *X, mxClassID computation_class )
{
    size_t  matrix_size     = 0;
    void   *matrix_buffer   = NULL;

#ifdef HAVE_OCTAVE
    int     n               = 0;
    int     lapack_status   = 0;
#else  /* MATLAB */
    ptrdiff_t n             = 0;
    ptrdiff_t lapack_status = 0;
#endif
    char    uplo            = 'U';
    int     is_complex_flag = 0;

    if( X == NULL || computation_class == mxUNKNOWN_CLASS )
        return NULL;

    n               = mxGetM( X );
    is_complex_flag = mxIsComplex( X );

    /* allocate a buffer for the inversion. */
    matrix_size = (n * n *
                   (is_complex_flag ? 2 : 1) *
                   (computation_class == mxDOUBLE_CLASS ? sizeof( double ) : sizeof( float )));
    if( NULL == (matrix_buffer = mxMalloc( matrix_size )) )
        mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:memoryAllocation",
                           "Failed to allocate %lu bytes for the inversion buffer.",
                           matrix_size );

    copy_split_to_interleaved( matrix_buffer,
                               mxGetPr( X ),
                               mxGetPi( X ),
                               n * n,
                               mxGetClassID( X ),
                               computation_class );

    /* do the Cholesky factorization */
    /* if X was not positive definite issue an error about it */
    if( computation_class == mxDOUBLE_CLASS )
    {
        /* factor */
        if( is_complex_flag )
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
        if( is_complex_flag )
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
        if( is_complex_flag )
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
        if( is_complex_flag )
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

    return matrix_buffer;
}

void copy_matrix( mxArray *X_inv, void *inverted_matrix, mxClassID computation_class )
{
    int       n           = 0;
    mxClassID X_inv_class = mxUNKNOWN_CLASS;

    if( X_inv == NULL || inverted_matrix == NULL ||
        !(computation_class == mxDOUBLE_CLASS ||
          computation_class == mxSINGLE_CLASS) )
        return;

    n           = mxGetM( X_inv );
    X_inv_class = mxGetClassID( X_inv );

    /* copy the data from contiguous to interleaved.  this properly handles
       mirroring the data from upper triangular to full matrix as well. */
    copy_interleaved_to_split( mxGetPr( X_inv ),
                               mxGetPi( X_inv ),
                               inverted_matrix,
                               n,
                               computation_class,
                               X_inv_class );

    return;
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    int       n            = 0;
    int       X_is_complex = 0;

    mxClassID X_class           = mxUNKNOWN_CLASS;
    mxClassID computation_class = mxUNKNOWN_CLASS;

    /* nonsense to avoid compiler warnings */
#ifdef HAVE_OCTAVE
    mwSize    output_dimensions[2];
#else
    size_t    output_dimensions[2];
#endif

    const mwSize   *input_dimensions = NULL;


    void     *inverted_matrix = NULL;

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
    X_class      = mxGetClassID( prhs[INPUT_X_INDEX] );
    X_is_complex = mxIsComplex( prhs[INPUT_X_INDEX] );

    /* verify that X is either single or double precision so we can simplify
       our lives. */
    switch( X_class )
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
        computation_class = X_class;

    if( NULL == (inverted_matrix = invert_matrix( prhs[INPUT_X_INDEX], computation_class )) )
    {
        mexWarnMsgIdAndTxt( "MATLAB:cholesky_inverse:warning",
                            "Failed to invert the matrix." );
        return;
    }

    /* if the user didn't want the output, return now. */
    if( nlhs < 1 )
    {
        mxFree( inverted_matrix );
        return;
    }

    /* allocate the output variable with the output precision and X's
       complexity.  since we'll immediately copy our interleaved matrix into
       this variable's buffer(s), we request uninitialized data. */
    input_dimensions = mxGetDimensions( prhs[INPUT_X_INDEX] );
    output_dimensions[0] = input_dimensions[0];
    output_dimensions[1] = input_dimensions[1];

#ifndef HAVE_OCTAVE
    if( NULL == (plhs[OUTPUT_X_INV_INDEX] = mxCreateUninitNumericArray( 2,
#else
    if( NULL == (plhs[OUTPUT_X_INV_INDEX] = mxCreateNumericArray(       2,
#endif
                                                                        output_dimensions,
                                                                        X_class,
                                                                        (X_is_complex ? mxCOMPLEX : mxREAL) )) )
    {
        mxFree( inverted_matrix );
        mexErrMsgIdAndTxt( "MATLAB:cholesky_inverse:memoryAllocation",
                           "Failed to allocate the output inverse matrix." );
    }

    /* copy the interleaved matrix into the return argument. */
    copy_matrix( plhs[OUTPUT_X_INV_INDEX], inverted_matrix, computation_class );

    mxFree( inverted_matrix );

    return;
}
