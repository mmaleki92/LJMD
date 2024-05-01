/* 
 * simple lennard-jones potential MD code with velocity verlet.
 * units: Length=Angstrom, Mass=amu; Energy=kcal
 *
 * optimization 1: apply newton's 3rd law
 * optimization 2: avoid using pow() in the force loop.
 * optimization 3: avoid using sqrt() in the force loop.
 * optimization 4: avoid division in the force loop.
 * optimization 5: move invariant expressions outside of loops.
 * optimization 6: use cell lists to reduce number of pairs to look at.
 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* generic file- or pathname buffer length */
#define BLEN 200

/* a few physical constants */
const double kboltz=0.0019872067;     /* boltzman constant in kcal/mol/K */
const double mvsq2e=2390.05736153349; /* m*v^2 in kcal/mol */

/* ratio between cutoff radius and length of a cell */
const double cellrat=2.0;
/* structure for cell-list data */
struct _cell {
    int natoms;                 /* number of atoms in this cell */
    int owner;                  /* task/thread id that owns this cell */
    int *idxlist;               /* list of atom indices */
};
typedef struct _cell cell_t;
    
/* structure to hold the complete information 
 * about the MD system */
struct _mdsys {
    double dt, mass, epsilon, sigma, box, rcut;
    double ekin, epot, temp, _pad1;
    double *rx, *ry, *rz;
    double *vx, *vy, *vz;
    double *fx, *fy, *fz;
    int mpirank, nsize;
    cell_t *clist;
    int *plist, _pad2;
    int natoms, nfi, nsteps;
    int ngrid, ncell, npair, nidx, _pad3;
    double delta;
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
static double pbc(double x, const double boxby2, const double box)
{
    while (x >  boxby2) x -= box;
    while (x < -boxby2) x += box;
    return x;
}

/* build and update cell list */
static void updcells(mdsys_t *sys)
{
    int i, j, k, m, n;
    int ngrid, ncell, npair, nidx, midx;
    double delta, boxby2, boxoffs;
    boxby2 = 0.5 * sys->box;
        
    if (sys->clist == NULL) {
        ngrid  = floor(cellrat * sys->box / sys->rcut);
        ncell  = ngrid*ngrid*ngrid;
        delta  = sys->box / ngrid;
        boxoffs= boxby2 - 0.5*delta;
        
        sys->delta = delta;
        sys->ngrid = ngrid;
        sys->ncell = ncell;

        /* allocate cell list storage */
        sys->clist = (cell_t *) malloc(ncell*sizeof(cell_t));
        sys->plist = (int *) malloc(2*ncell*ncell*sizeof(int));

        /* allocate index lists within cell. cell density < 2x avg. density */
        nidx = 2*sys->natoms / ncell + 2;
        nidx = ((nidx/2) + 1) * 2;
        sys->nidx = nidx;
        for (i=0; i<ncell; ++i) {
            sys->clist[i].idxlist = (int *) malloc(nidx*sizeof(int));
        }

        /* build cell pair list, assuming newtons 3rd law. */
        npair = 0;
        for (i=0; i < ncell-1; ++i) {
            double x1,x2,y1,y2,z1,z2,rx,ry,rz;
            
            k  = i/ngrid/ngrid;
            x1 = k*delta - boxoffs;
            y1 = ((i-(k*ngrid*ngrid))/ngrid)*delta - boxoffs;
            z1 = (i % ngrid)*delta - boxoffs;

            for (j=i+1; j<ncell; ++j) {
                k  = j/ngrid/ngrid;
                x2 = k*delta - boxoffs;
                y2 = ((j-(k*ngrid*ngrid))/ngrid)*delta - boxoffs;
                z2 = (j % ngrid)*delta - boxoffs;

                rx=pbc(x1 - x2, boxby2, sys->box);
                ry=pbc(y1 - y2, boxby2, sys->box);
                rz=pbc(z1 - z2, boxby2, sys->box);

                /* check for cells on a line that are too far apart */
                if (fabs(rx) > sys->rcut+delta) continue;
                if (fabs(ry) > sys->rcut+delta) continue;
                if (fabs(rz) > sys->rcut+delta) continue;

                /* check for cells in a plane that are too far apart */
                if (sqrt(rx*rx+ry*ry) > (sys->rcut+sqrt(2.0)*delta)) continue;
                if (sqrt(rx*rx+rz*rz) > (sys->rcut+sqrt(2.0)*delta)) continue;
                if (sqrt(ry*ry+rz*rz) > (sys->rcut+sqrt(2.0)*delta)) continue;

                /* other cells that are too far apart */
                if (sqrt(rx*rx + ry*ry + rz*rz) > (sqrt(3.0)*delta+sys->rcut)) continue;
                
                /* cells are close enough. add to list */
                sys->plist[2*npair  ] = i;
                sys->plist[2*npair+1] = j;
                ++npair;
            }
        }
        sys->npair = npair;
        printf("Cell list has %dx%dx%d=%d cells with %d/%d pairs and "
               "%d atoms/celllist.\n", ngrid, ngrid, ngrid, sys->ncell, 
               sys->npair, ncell*(ncell-1)/2, nidx);
    }

    /* reset cell list and sort atoms into cells */
    ncell=sys->ncell;
    delta=sys->delta;
    ngrid=sys->ngrid;
    
    for (i=0; i < sys->ncell; ++i) {
        sys->clist[i].natoms=0;
    }

    boxoffs= boxby2 - 0.5*delta;
    midx=0;
    for (i=0; i < sys->natoms; ++i) {
        int idx;
        
        k=floor((pbc(sys->rx[i], boxby2, sys->box)+boxby2)/delta);
        m=floor((pbc(sys->ry[i], boxby2, sys->box)+boxby2)/delta);
        n=floor((pbc(sys->rz[i], boxby2, sys->box)+boxby2)/delta);
        j = ngrid*ngrid*k+ngrid*m+n;

        idx = sys->clist[j].natoms;
        sys->clist[j].idxlist[idx]=i;
        ++idx;
        sys->clist[j].natoms = idx;
        if (idx > midx) midx=idx;
    }
    if (midx > sys->nidx) {
        printf("overflow in cell list: %d/%d atoms/cells.\n", midx, sys->nidx);
        exit(1);
    }
    return;
}


/* release cell list storage */
static void free_cell_list(mdsys_t *sys)
{
    int i;
    
    if (sys->clist == NULL) 
        return;
    
    for (i=0; i < sys->ncell; ++i) {
        free(sys->clist[i].idxlist);
    }
    
    free(sys->clist);
    sys->clist = NULL;
    sys->ncell = 0;
}


/* compute kinetic energy */
static void ekin(mdsys_t *sys)
{   
    int i;
    
    sys->ekin=0.0;
    for (i=0; i<sys->natoms; ++i) {
        sys->ekin += sys->vx[i]*sys->vx[i] 
            + sys->vy[i]*sys->vy[i] 
            + sys->vz[i]*sys->vz[i];
    }
    sys->ekin *= 0.5*mvsq2e*sys->mass;
    sys->temp  = 2.0*sys->ekin/(3.0*sys->natoms-3.0)/kboltz;
}

/* compute forces */
static void force(mdsys_t *sys) 
{
    double ffac,c12,c6,boxby2;
    double rx,ry,rz,rsq,rcsq;
    int i,j,k;

    /* zero energy and forces */
    sys->epot=0.0;
    azzero(sys->fx,sys->natoms);
    azzero(sys->fy,sys->natoms);
    azzero(sys->fz,sys->natoms);

    /* precompute some constants */
    c12 = 4.0*sys->epsilon*pow(sys->sigma,12.0);
    c6  = 4.0*sys->epsilon*pow(sys->sigma, 6.0);
    rcsq= sys->rcut * sys->rcut;
    boxby2 = 0.5*sys->box;

    /* self interaction of atoms in cell */
    for(i=0; i < sys->ncell; ++i) {
        const cell_t *c1;
        
        c1=sys->clist + i;
        for (j=0; j < c1->natoms-1; ++j) {
            int ii, jj;
            double rx1, ry1, rz1;

            ii=c1->idxlist[j];
            rx1=sys->rx[ii];
            ry1=sys->ry[ii];
            rz1=sys->rz[ii];
        
            for(k=j+1; k < c1->natoms; ++k) {
                jj=c1->idxlist[k];
                
                /* get distance between particle i and j */
                rx=pbc(rx1 - sys->rx[jj], boxby2, sys->box);
                ry=pbc(ry1 - sys->ry[jj], boxby2, sys->box);
                rz=pbc(rz1 - sys->rz[jj], boxby2, sys->box);
                rsq = rx*rx + ry*ry + rz*rz;
                
                /* compute force and energy if within cutoff */
                if (rsq < rcsq) {
                    double r6,rinv;

                    rinv=1.0/rsq;
                    r6=rinv*rinv*rinv;
                    
                    ffac = (12.0*c12*r6 - 6.0*c6)*r6*rinv;
                    sys->epot += r6*(c12*r6 - c6);

                    sys->fx[ii] += rx*ffac;
                    sys->fy[ii] += ry*ffac;
                    sys->fz[ii] += rz*ffac;
                    sys->fx[jj] -= rx*ffac;
                    sys->fy[jj] -= ry*ffac;
                    sys->fz[jj] -= rz*ffac;
                }
            }
        }
    }    

    /* interaction of atoms in different cells */
    for(i=0; i < sys->npair; ++i) {
        const cell_t *c1, *c2;
        
        c1=sys->clist + sys->plist[2*i];
        c2=sys->clist + sys->plist[2*i+1];
        
        for (j=0; j < c1->natoms; ++j) {
            int ii, jj;
            double rx1, ry1, rz1;

            ii=c1->idxlist[j];
            rx1=sys->rx[ii];
            ry1=sys->ry[ii];
            rz1=sys->rz[ii];
        
            for(k=0; k < c2->natoms; ++k) {
                jj=c2->idxlist[k];
                
                /* get distance between particle i and j */
                rx=pbc(rx1 - sys->rx[jj], boxby2, sys->box);
                ry=pbc(ry1 - sys->ry[jj], boxby2, sys->box);
                rz=pbc(rz1 - sys->rz[jj], boxby2, sys->box);
                rsq = rx*rx + ry*ry + rz*rz;
                
                /* compute force and energy if within cutoff */
                if (rsq < rcsq) {
                    double r6,rinv;

                    rinv=1.0/rsq;
                    r6=rinv*rinv*rinv;
                    
                    ffac = (12.0*c12*r6 - 6.0*c6)*r6*rinv;
                    sys->epot += r6*(c12*r6 - c6);

                    sys->fx[ii] += rx*ffac;
                    sys->fy[ii] += ry*ffac;
                    sys->fz[ii] += rz*ffac;
                    sys->fx[jj] -= rx*ffac;
                    sys->fy[jj] -= ry*ffac;
                    sys->fz[jj] -= rz*ffac;
                }
            }
        }
    }
}

/* velocity verlet */
static void velverlet(mdsys_t *sys)
{
    int i;
    double dtmf;
    dtmf = 0.5*sys->dt / mvsq2e / sys->mass;

    /* first part: propagate velocities by half and positions by full step */
    for (i=0; i<sys->natoms; ++i) {
        sys->vx[i] += dtmf * sys->fx[i];
        sys->vy[i] += dtmf * sys->fy[i];
        sys->vz[i] += dtmf * sys->fz[i];
        sys->rx[i] += sys->dt*sys->vx[i];
        sys->ry[i] += sys->dt*sys->vy[i];
        sys->rz[i] += sys->dt*sys->vz[i];
    }

    /* compute forces and potential energy */
    force(sys);

    /* second part: propagate velocities by another half step */
    for (i=0; i<sys->natoms; ++i) {
        sys->vx[i] += dtmf * sys->fx[i];
        sys->vy[i] += dtmf * sys->fy[i];
        sys->vz[i] += dtmf * sys->fz[i];
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

    int nprint, i;
    char restfile[BLEN], trajfile[BLEN], ergfile[BLEN], line[BLEN];
    FILE *fp,*traj,*erg;
    mdsys_t sys;

    MPI_Comm_size( MPI_COMM_WORLD, &sys.nsize);
    MPI_Comm_rank( MPI_COMM_WORLD, &sys.mpirank);
    sys.mpicomm = MPI_COMM_WORLD;


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
            azzero(sys.fx, sys.natoms);
            azzero(sys.fy, sys.natoms);
            azzero(sys.fz, sys.natoms);
        } else {
            perror("cannot read restart file");
            return 3;
        }
    }

    if (sys.mpirank == 0) {

        erg=fopen(ergfile,"w");
        traj=fopen(trajfile,"w");
    }

    /* create initial cell list */
    sys.clist = NULL;
    sys.plist = NULL;
    updcells(&sys);

    /* initialize forces and energies.*/
    sys.nfi=0;
    force(&sys);
    ekin(&sys);
    
    printf("Starting simulation with %d atoms for %d steps.\n",sys.natoms, sys.nsteps);
    printf("     NFI            TEMP            EKIN                 EPOT              ETOT\n");
    output(&sys, erg, traj);

    /**************************************************/
    /* main MD loop */
    for(sys.nfi=1; sys.nfi <= sys.nsteps; ++sys.nfi) {

        /* write output, if requested */
        if ((sys.nfi % nprint) == 0) {
            output(&sys, erg, traj);
        }

        /* propagate system and recompute energies */
        velverlet(&sys);
        ekin(&sys);

        /* update cell list */
        updcells(&sys);
    }
    /**************************************************/

    if (sys.mpirank == 0){

        /* clean up: close files, free memory */
        fclose(erg);
        fclose(traj);
    }
    
    free(sys.rx);
    free(sys.ry);
    free(sys.rz);
    free(sys.vx);
    free(sys.vy);
    free(sys.vz);
    free(sys.fx);
    free(sys.fy);
    free(sys.fz);
    free_cell_list(&sys);
    printf("Simulation Done.\n");
    MPI_Finalize();

    return 0;
}
