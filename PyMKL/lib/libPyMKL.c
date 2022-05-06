/*========================================================*
 * libPyMKL.c
 *
 * The calling syntax from Python is:
 *
 *		(S_W_A , S_D_A) = computeSWA(K_tot_ALL, A, W, Diag);
 *		(S_W_B , S_D_B) = computeSWB(K_tot_ALL, betas, W, Diag);
 *		gap = computeENERGY(K_tot_ALL,betas,A,W);
 *
 *========================================================*/

#include <stdlib.h>
#include <Python.h>
#include <stdio.h>

/* The computational routine */
void computeSWA(double *K_tot_ALL, double *A, double *W, double *Diag, double *S_W_A, double *S_D_A, int N, int NA, int m, int i1, int i2)
{
    int r,l,i,j,c,c1,c2,k;

    double *tmp = malloc(sizeof(double)*m*NA);
    double *KiA = malloc(sizeof(double)*N*m*NA);

    double tmpV;
    
    for(i=0; i<N; i++)
    {
        for(c=0; c<m; c++)
        {
            for(r=0; r<NA; r++)
            {
                tmpV = 0.;
                for(l=0; l<N; l++)
                {
                    tmpV += K_tot_ALL[i + l*N + c*N*N] * A[l + r*N];
                }
                KiA[i + N*c + r*N*m] = tmpV;
            }
        }
    }
    
    for(i=i1; i<i2; i++)
    {
        for(j=(i+1); j<N; j++)
        {
            const double Wij = W[i + j*N];
            
            if(Wij != 0.)
            {
                for(c1=0; c1<m; c1++)
                {
                    for(c2=c1; c2<m; c2++) 
                    {
                        tmpV = 0.;
                        for(k=0; k<NA; k++)
                        {
                            double tmp1 = KiA[i + N*c1 + k*N*m] - KiA[j + N*c1 + k*N*m];
                            double tmp2 = KiA[i + N*c2 + k*N*m] - KiA[j + N*c2 + k*N*m];
                            tmpV += tmp1 * tmp2;
                        }
                        S_W_A[c1 + c2*m] += 2. * Wij * tmpV;
                    }
                }
            }
        }
                
        
        for(c=0; c<m; c++)
        {
            for(r=0; r<NA; r++)
            {
                tmpV = 0.;
                for(l=0; l<N; l++)
                {
                    tmpV += K_tot_ALL[i + l*N + c*N*N] * A[l + r*N];
                }
                tmp[c + r*m] = tmpV;
            }
        }        
        
        const double Diagii = Diag[i];

        for(c1=0; c1<m; c1++)
        {
            for(c2=c1; c2<m; c2++) 
            {
                tmpV = 0.;
                for(k=0; k<NA; k++)
                {
                    tmpV += (tmp[c1 + k*m] * tmp[c2 + k*m]);
                }
                S_D_A[c1 + c2*m] += Diagii * tmpV;
            }
        }        
    }
    free(tmp);
    free(KiA);
}

/* The computational routine */
void computeSWB(double *K_tot_ALL, double *betas, double *W, double *Diag, double *S_W_B, double *S_D_B, int N, int m, int i1, int i2)
{
    int r,l,i,j,c;

    
    double *tmp = malloc(sizeof(double)*N);
    
    double tmpV;
    
    for(i=i1; i<i2; i++)
    {
        for(j=(i+1); j<N; j++)
        {
            const double Wij = W[i + j*N];
            
            if(Wij != 0.)
            {
                for(r=0; r<N; r++)
                {
                    tmpV = 0.;
                    for(c=0; c<m; c++)
                    {
                        tmpV += (K_tot_ALL[i + r*N + c*N*N] - K_tot_ALL[j + r*N + c*N*N]) * betas[c];
                    }
                    tmp[r] = tmpV;
                }

                for(r=0; r<N; r++)
                {
                    for(l=r; l<N; l++)
                    {
                        S_W_B[r + l*N] += 2. * Wij * (tmp[r] * tmp[l]);
                    }
                }
            }
        }
        
        
        for(r=0; r<N; r++)
        {
            tmpV = 0.;
            for(c=0; c<m; c++)
            {
                tmpV += K_tot_ALL[i + r*N + c*N*N] * betas[c];                
            }
            tmp[r] = tmpV;
        }        
        
        const double Diagii = Diag[i];

        tmpV = 0.;
        for(r=0; r<N; r++)
        {
            for(l=r; l<N; l++) 
            {
                S_D_B[r + l*N] += Diagii * (tmp[r] * tmp[l]);
            }
        }
    }
    free(tmp);
}

/* The computational routine */
void computeENERGY(double *K_tot_ALL, double *betas, double *A, double *W, double *Diag, int N, int NA, int m, double output[2], int i1, int i2)
{
    int r,l,i,j,c;

    double gap = 0.;
    double constr = 0.;

    double *tmp = malloc(sizeof(double)*NA);
    double *tmpM = malloc(sizeof(double)*N*N);
    
    double tmpV;

    /* precompute K*betas*/
    for(i=0; i<N; i++)
    {
        for(r=0; r<N; r++)
        {
            tmpV = 0.;
            for(c=0; c<m; c++)
            {
                tmpV += K_tot_ALL[i + r*N + c*N*N] * betas[c];                
            }
            tmpM[i + r*N] = tmpV;
        }
    }
    
    for(i=i1; i<i2; i++)
    {
        for(j=(i+1); j<N; j++)
        {        
            const double Wij = W[i + j*N];
            
            if(Wij != 0.)
            {
                for(r=0; r<NA; r++)
                {
                    tmpV = 0.;
                    for(l=0; l<N; l++)
                    {
                        tmpV += A[l + r*N] * (tmpM[i + l*N] - tmpM[j + l*N]);
                    }
                    tmp[r] = tmpV;
                }

                double gapij = 0.;
                for(r=0; r<NA; r++)
                {
                    gapij += tmp[r] * tmp[r];
                }

                gap += 2. * gapij * Wij;   
            }
        }
        
        for(r=0; r<NA; r++)
        {
            tmpV = 0.;
            for(l=0; l<N; l++)
            {
                tmpV += A[l + r*N] * tmpM[i + l*N];                 
            }
            tmp[r] = tmpV;
        }
        double constrii = 0.;
        for(r=0; r<NA; r++)
        {
            constrii += tmp[r] * tmp[r];
        }
        constr += constrii * Diag[i];            
    }
    free(tmp);
    free(tmpM);
    
    output[0] = gap;
    output[1] = constr;
}

