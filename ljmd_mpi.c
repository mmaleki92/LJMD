/* 
 * simple lennard-jones potential MD code with velocity verlet.
 * units: Length=Angstrom, Mass=amu; Energy=kcal
 *
 * baseline c version.
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>


/* generic file- or pathname buffer length */
#define BLEN 200

/* a few physical constants */
const double kboltz=0.0019872067;     /* boltzman constant in kcal/mol/K */
const double mvsq2e=2390.05736153349; /* m*v^2 in kcal/mol */

/* structure to hold the complete information 
 * about the MD system */
struct _mdsys {
    int natoms,nfi,nsteps;
    int nsize;
    int mpirank; 
    double dt, mass, epsilon, sigma, box, rcut;
    double ekin, epot, temp;
    double *rx, *ry, *rz;
    double *vx, *vy, *vz;
    double *fx, *fy, *fz;
    double *cx, *cy, *cz;
    MPI_Comm mpicomm;
};
typedef struct _mdsys mdsys_t;

/* helper function: read a line and then return
   the first string with whitespace stripped off */
static int get_a_line(FILE *fp, char *buf)
{
    char tmp[BLEN], *ptr;

    /* read a line and cut of comments and blanks */
    if (fgets(tmp,BLEN,fp)) {
        int i;

        ptr=strchr(tmp,'#');
        if (ptr) *ptr= '\0';
        i=strlen(tmp); --i;
        while(isspace(tmp[i])) {
            tmp[i]='\0';
            --i;
        }
        ptr=tmp;
        while(isspace(*ptr)) {++ptr;}
        i=strlen(ptr);
        strcpy(buf,tmp);
        return 0;
    } else {
        perror("problem reading input");
        return -1;
    }
    return 0;
}
 
/* helper function: zero out an array */
static void azzero(double *d, const int n)
{
    int i;
    for (i=0; i<n; ++i) {
        *d++=0.0;
    }
}

/* helper function: apply minimum image convention */
static double pbc(double x, const double boxby2) {
    int count = 0;  // safeguard against infinite loops
    while (x > boxby2 && count < 100) {
        x -= boxby2 * 2;
        ++count;
    }
    while (x < -boxby2 && count < 100) {
        x += boxby2 * 2;
        ++count;
    }
    if (count >= 100) {
        fprintf(stderr, "pbc function stuck: x=%f, boxby2=%f\n", x, boxby2);
    }
    return x;
}


/* compute kinetic energy */
static void ekin(mdsys_t *sys)
{   
    int i;
    
    sys->ekin=0.0;
    for (i=0; i<sys->natoms; ++i) {
        sys->ekin += 0.5*mvsq2e*sys->mass*(sys->vx[i]*sys->vx[i] + sys->vy[i]*sys->vy[i] + sys->vz[i]*sys->vz[i]);
    }
    sys->temp = 2.0*sys->ekin/(3.0*sys->natoms-3.0)/kboltz;
}

static void force(mdsys_t *sys) 
{
    // azzero(sys->fx,sys->natoms);
    // azzero(sys->fy,sys->natoms);
    // azzero(sys->fz,sys->natoms);
    double r, ffac;
    double rx, ry, rz;
    int ii, i, j;
    double epot=0.0;
    double rsq, rcsq, c6, c12;
    // Zeroing the accumulators
    // sys->epot = 0.0;
    azzero(sys->cx, sys->natoms);
    azzero(sys->cy, sys->natoms);
    azzero(sys->cz, sys->natoms);
    
    MPI_Bcast(sys->rx, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    MPI_Bcast(sys->ry, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    MPI_Bcast(sys->rz, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    
    MPI_Bcast(sys->vx, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    MPI_Bcast(sys->vy, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    MPI_Bcast(sys->vz, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    
    // Calculate the range of particles this process will handle
    
    c12=4.0*sys->epsilon*pow(sys->sigma,12.0);
    c6 =4.0*sys->epsilon*pow(sys->sigma, 6.0);
    rcsq = sys->rcut * sys->rcut;

    int local_start = sys->mpirank * (sys->natoms / sys->nsize);
    int local_end = (sys->mpirank+1) * (sys->natoms / sys->nsize);
    if (sys->mpirank == sys->nsize - 1) {
        local_end = sys->natoms;  // Ensure the last process covers all remaining atoms
    }

    //TODO: Problem? 

    // #pragma omp parallel for private(j, rx, ry, rz, rsq, r, ffac) reduction(+:epot) 
    // for (i=0; i < sys->natoms-1; i += sys->nsize) {
    //         ii = i + sys->mpirank;
    //     if (ii >= (sys->natoms - 1)) break;

    #if defined(_OPENMP)
    #pragma omp parallel for private(j, rx, ry, rz, rsq, r, ffac) reduction(+:epot) 
    #endif
    for (i = local_start; i < local_end; ++i) {
        for (j = 0; j < sys->natoms; ++j) {
            if (i != j) {
                

            rx=pbc(sys->rx[i] - sys->rx[j], 0.5*sys->box);
            ry=pbc(sys->ry[i] - sys->ry[j], 0.5*sys->box);
            rz=pbc(sys->rz[i] - sys->rz[j], 0.5*sys->box);
            rsq = rx*rx + ry*ry + rz*rz;

            if (rsq < rcsq) {
                double r6,rinv; rinv=1.0/rsq; r6=rinv*rinv*rinv;
                ffac = (12.0*c12*r6 - 6.0*c6)*r6*rinv;
                epot += 0.5 * r6*(c12*r6 - c6);

                sys->cx[i] += rx*ffac; sys->cx[j] -= rx*ffac;
                sys->cy[i] += ry*ffac; sys->cy[j] -= ry*ffac;
                sys->cz[i] += rz*ffac; sys->cz[j] -= rz*ffac;
                
            }
            }
        }
    }

    MPI_Reduce(sys->cx, sys->fx, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    MPI_Reduce(sys->cy, sys->fy, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    MPI_Reduce(sys->cz, sys->fz, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    MPI_Reduce(&epot, &sys->epot, 1, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    
}

/* velocity verlet */
static void velverlet(mdsys_t *sys)
{
    int i;

    /* first part: propagate velocities by half and positions by full step */
    for (i=0; i<sys->natoms; ++i) {
        sys->vx[i] += 0.5*sys->dt / mvsq2e * sys->fx[i] / sys->mass;
        sys->vy[i] += 0.5*sys->dt / mvsq2e * sys->fy[i] / sys->mass;
        sys->vz[i] += 0.5*sys->dt / mvsq2e * sys->fz[i] / sys->mass;
        sys->rx[i] += sys->dt*sys->vx[i];
        sys->ry[i] += sys->dt*sys->vy[i];
        sys->rz[i] += sys->dt*sys->vz[i];
    }

    /* compute forces and potential energy */
    force(sys);

    /* second part: propagate velocities by another half step */
    for (i=0; i<sys->natoms; ++i) {
        sys->vx[i] += 0.5*sys->dt / mvsq2e * sys->fx[i] / sys->mass;
        sys->vy[i] += 0.5*sys->dt / mvsq2e * sys->fy[i] / sys->mass;
        sys->vz[i] += 0.5*sys->dt / mvsq2e * sys->fz[i] / sys->mass;
    }
}


/* append data to output. */
static void output(mdsys_t *sys, FILE *erg, FILE *traj)
{
    int i;
    
    printf("% 8d % 20.8f % 20.8f % 20.8f % 20.8f\n", sys->nfi, sys->temp, sys->ekin, sys->epot, sys->ekin+sys->epot);
    fprintf(erg,"% 8d % 20.8f % 20.8f % 20.8f % 20.8f\n", sys->nfi, sys->temp, sys->ekin, sys->epot, sys->ekin+sys->epot);
    fprintf(traj,"%d\n nfi=%d etot=%20.8f\n", sys->natoms, sys->nfi, sys->ekin+sys->epot);
    for (i=0; i<sys->natoms; ++i) {
        fprintf(traj, "Ar  %20.8f %20.8f %20.8f\n", sys->rx[i], sys->ry[i], sys->rz[i]);
    }
}


/* main */
int main(int argc, char **argv) 
{
    // MPI Initialization phase
    MPI_Init( &argc, &argv);

    mdsys_t sys;

    MPI_Comm_size( MPI_COMM_WORLD, &sys.nsize);
    MPI_Comm_rank( MPI_COMM_WORLD, &sys.mpirank);
    sys.mpicomm = MPI_COMM_WORLD;
    
    #if defined(_OPENMP)
    omp_set_num_threads(2);  // Example: set to 4 threads
    #endif
    
    int nprint, i;
    char restfile[BLEN], trajfile[BLEN], ergfile[BLEN], line[BLEN];
    
    FILE *fp,*traj,*erg;
   
    
    if (sys.mpirank == 0) {
        /* read input file */
        if(get_a_line(stdin,line)) return 1;
        sys.natoms=atoi(line);
        
        if(get_a_line(stdin,line)) return 1;
        sys.mass=atof(line);        
        
        if(get_a_line(stdin,line)) return 1;
        sys.epsilon=atof(line);

        if(get_a_line(stdin,line)) return 1;
        sys.sigma=atof(line);

        if(get_a_line(stdin,line)) return 1;
        sys.rcut=atof(line);

        if(get_a_line(stdin,line)) return 1;
        sys.box=atof(line);

        if(get_a_line(stdin,restfile)) return 1;
        if(get_a_line(stdin,trajfile)) return 1;
        if(get_a_line(stdin,ergfile)) return 1;
        if(get_a_line(stdin,line)) return 1;
        sys.nsteps=atoi(line);
        
        if(get_a_line(stdin,line)) return 1;
        sys.dt=atof(line);
        
        if(get_a_line(stdin,line)) return 1;
        nprint=atoi(line);
    }

    MPI_Bcast(&sys.natoms, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.rcut, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.box, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.nsteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* allocate memory */
    sys.rx=(double *)malloc(sys.natoms*sizeof(double));
    sys.ry=(double *)malloc(sys.natoms*sizeof(double));
    sys.rz=(double *)malloc(sys.natoms*sizeof(double));
    
    sys.vx=(double *)malloc(sys.natoms*sizeof(double));
    sys.vy=(double *)malloc(sys.natoms*sizeof(double));
    sys.vz=(double *)malloc(sys.natoms*sizeof(double));

    sys.fx=(double *)malloc(sys.natoms*sizeof(double));
    sys.fy=(double *)malloc(sys.natoms*sizeof(double));
    sys.fz=(double *)malloc(sys.natoms*sizeof(double));

    sys.cx=(double *)malloc(sys.natoms*sizeof(double));
    sys.cy=(double *)malloc(sys.natoms*sizeof(double));
    sys.cz=(double *)malloc(sys.natoms*sizeof(double)); 


    if (sys.mpirank == 0) {
        /* read restart */
        fp=fopen(restfile,"r");
        if(fp) {
            for (i=0; i<sys.natoms; ++i) {
                fscanf(fp,"%lf%lf%lf",sys.rx+i, sys.ry+i, sys.rz+i);
            }
            for (i=0; i<sys.natoms; ++i) {
                fscanf(fp,"%lf%lf%lf",sys.vx+i, sys.vy+i, sys.vz+i);
            }
            fclose(fp);
        

        } else {
            perror("cannot read restart file");
            return 3;
        }

    }

    sys.nfi=0;

    /* initialize forces and energies.*/
    force(&sys); 
    ekin(&sys);

    if (sys.mpirank == 0) {
        erg=fopen(ergfile,"w");
        traj=fopen(trajfile,"w");
        printf("Starting simulation with %d atoms for %d steps.\n",sys.natoms, sys.nsteps);
        printf("     NFI            TEMP            EKIN                 EPOT              ETOT\n");
        output(&sys, erg, traj);
    }


    /**************************************************/
    /* main MD loop */
    for(sys.nfi=1; sys.nfi <= sys.nsteps; ++sys.nfi) {

        if (sys.mpirank == 0) {
        /* write output, if requested */
        if ((sys.nfi % nprint) == 0)
            output(&sys, erg, traj);
        }

        /* propagate system and recompute energies */
        velverlet(&sys);
        ekin(&sys);
    }
    // /**************************************************/


    free(sys.rx);
    free(sys.ry);
    free(sys.rz);
    free(sys.vx);
    free(sys.vy);
    free(sys.vz);
    free(sys.fx);
    free(sys.fy);
    free(sys.fz);

    if (sys.mpirank == 0){
        /* clean up: close files, free memory */
        fclose(erg);
        fclose(traj);
        printf("Simulation Done.\n");
    }


    MPI_Finalize();

    return 0;
}
