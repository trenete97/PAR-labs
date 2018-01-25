#include "heat.h"
#include "omp.h"

/*
 * Function to copy one matrix into another
 */

void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey)
{
    #pragma omp parallel for
    for (int i=1; i<=sizex-2; i++)
        for (int j=1; j<=sizey-2; j++) 
            v[ i*sizey+j ] = u[ i*sizey+j ];
}

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
  
    int howmany=omp_get_max_threads();
    #pragma omp parallel for private(diff) reduction(+:sum)
    for (int blockid = 0; blockid < howmany; ++blockid) {
      int i_start = lowerb(blockid, howmany, sizex);
      int i_end = upperb(blockid, howmany, sizex);
      for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
        for (int j=1; j<= sizey-2; j++) {
	     utmp[i*sizey+j]= 0.25 * ( u[ i*sizey     + (j-1) ]+  // left
	                               u[ i*sizey     + (j+1) ]+  // right
				       u[ (i-1)*sizey + j     ]+  // top
				       u[ (i+1)*sizey + j     ]); // bottom
	     diff = utmp[i*sizey+j] - u[i*sizey + j];
	     sum += diff * diff; 
	 }
      }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;

    int howmany=omp_get_max_threads();
    int howmanyc = 4;

    #pragma omp parallel for schedule(static) ordered (2) private(diff) reduction(+:sum)
    for (int blockidf = 0; blockidf < howmany; ++blockidf) {
        for(int blockidc = 0; blockidc < howmanyc; ++blockidc) {
            int j_start = lowerb(blockidc, howmanyc, sizey);
            int j_end = upperb(blockidc, howmanyc, sizey);
            int i_start = lowerb(blockidf, howmany, sizex);
            int i_end = upperb(blockidf, howmany, sizex);
            #pragma omp ordered depend(sink: blockidf-1, blockidc) depend(sink: blockidf, blockidc-1)
            for (int i=max(1, i_start); i<= min(sizex-2, i_end); i++) {
                for (int j=max(1, j_start); j<= min(sizey-2, j_end); j++) {
                    unew= 0.25 * ( u[ i*sizey	+ (j-1) ]+  // left
                    u[ i*sizey + (j+1)]+  // right
                    u[ (i-1)*sizey + j]+  // top
                    u[ (i+1)*sizey + j]); // bottom
                    diff = unew - u[i*sizey+ j];
                    sum += diff * diff; 
                    u[i*sizey+j]=unew;
                }
            }
            #pragma omp ordered depend(source)
        }
    }

    return sum;
}
