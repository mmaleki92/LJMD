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
/* Define constants for directions for easier reference */
#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

/* a few physical constants */
const double kboltz=0.0019872067;     /* boltzman constant in kcal/mol/K */
const double mvsq2e=2390.05736153349; /* m*v^2 in kcal/mol */

/* ratio between cutoff radius and length of a cell */
const double cellrat=1.0;
/* structure for cell-list data */
struct _cell {
    int natoms;                 /* number of atoms in this cell */
    int owner;                  /* task/thread id that owns this cell */
    int *idxlist;               /* list of atom indices */
    int nghosts;                /* number of ghost atoms */
    int *ghostlist;             /* list of ghost atom indices */
};
typedef struct _cell cell_t;

typedef struct {
    double rx, ry, rz;    // Positions
    double vx, vy, vz;    // Velocities
} particle_data;


/* structure to hold the complete information 
 * about the MD system */
struct _mdsys {
    double dt, mass, epsilon, sigma, box, rcut;
    double ekin, epot, temp, _pad1;
    double *rx, *ry, *rz;
    double *vx, *vy, *vz;
    double *fx, *fy, *fz;
    double *cx, *cy, *cz;
    int mpirank, nsize;
    cell_t *clist;
    int start_cell;
    int num_boundary_atoms;
    double *boundary_atoms;
    int end_cell;
    int *plist, _pad2;
    int natoms, nfi, nsteps;
    int ngrid, ncell, npair, nidx, _pad3;
    double delta;
    MPI_Comm mpicomm;
};
typedef struct _mdsys mdsys_t;

void free_resources(mdsys_t *sys) {
    free(sys->rx); free(sys->ry); free(sys->rz);
    free(sys->vx); free(sys->vy); free(sys->vz);
    free(sys->fx); free(sys->fy); free(sys->fz);
    free(sys->clist);
    free(sys->plist);
}

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
void setup_neighbors(mdsys_t *sys, int *neighbors) {
    int dims[1], periods[1], coords[1];
    MPI_Comm newcomm;

    dims[0] = sys->nsize;  // Total number of processes in a single dimension
    periods[0] = 1;        // Periodic boundary conditions

    // Create a 1D Cartesian topology
    if (MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 1, &newcomm) != MPI_SUCCESS) {
        fprintf(stderr, "Failed to create 1D Cartesian communicator.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    sys->mpicomm = newcomm; // Use the new communicator for subsequent operations

    // Get the coordinates in this new topology
    MPI_Cart_coords(sys->mpicomm, sys->mpirank, 1, coords);
    MPI_Cart_shift(sys->mpicomm, 0, 1, &neighbors[WEST], &neighbors[EAST]);  // Get neighbors in the linear layout
}

/* build and update cell list */
static void updcells(mdsys_t *sys)
{
    int i, j, k, m, n, idx, cid;
    int ngrid, ncell, npair, nidx, midx;
    double delta, boxby2, boxoffs;
    boxby2 = 0.5 * sys->box;
        
    // if (sys->clist == NULL) {
    //     ngrid  = floor(cellrat * sys->box / sys->rcut);
    //     ncell  = ngrid*ngrid*ngrid;
    //     delta  = sys->box / ngrid;
    //     boxoffs= boxby2 - 0.5*delta;
        
    //     sys->delta = delta;
    //     sys->ngrid = ngrid;
    //     sys->ncell = ncell;
    
    //     /* allocate cell list storage */
    //     sys->clist = (cell_t *) malloc(ncell*sizeof(cell_t));
    //     // sys->plist = (int *) malloc(2*ncell*ncell*sizeof(int));

    //     /* allocate index lists within cell. cell density < 2x avg. density */
    //     nidx = 2*sys->natoms / ncell + 2;
    //     nidx = ((nidx/2) + 1) * 2;
    //     sys->nidx = nidx;
    //     for (i=0; i<ncell; ++i) {
    //         sys->clist[i].idxlist = (int *) malloc(nidx*sizeof(int));
    //         sys->clist[i].natoms = 0;  // Ensure natoms is initialized to zero
    //     }

    //     /* build cell pair list, assuming newtons 3rd law. */
    //     // npair = 0;
    //     // for (i=0; i < ncell-1; ++i) {
    //     //     double x1,x2,y1,y2,z1,z2,rx,ry,rz;
            
    //     //     k  = i/ngrid/ngrid;
    //     //     x1 = k*delta - boxoffs;
    //     //     y1 = ((i-(k*ngrid*ngrid))/ngrid)*delta - boxoffs;
    //     //     z1 = (i % ngrid)*delta - boxoffs;

    //     //     for (j=i+1; j<ncell; ++j) {
    //     //         k  = j/ngrid/ngrid;
    //     //         x2 = k*delta - boxoffs;
    //     //         y2 = ((j-(k*ngrid*ngrid))/ngrid)*delta - boxoffs;
    //     //         z2 = (j % ngrid)*delta - boxoffs;

    //     //         rx=pbc(x1 - x2, boxby2, sys->box);
    //     //         ry=pbc(y1 - y2, boxby2, sys->box);
    //     //         rz=pbc(z1 - z2, boxby2, sys->box);

    //     //         /* check for cells on a line that are too far apart */
    //     //         if (fabs(rx) > sys->rcut+delta) continue;
    //     //         if (fabs(ry) > sys->rcut+delta) continue;
    //     //         if (fabs(rz) > sys->rcut+delta) continue;

    //     //         /* check for cells in a plane that are too far apart */
    //     //         if (sqrt(rx*rx+ry*ry) > (sys->rcut+sqrt(2.0)*delta)) continue;
    //     //         if (sqrt(rx*rx+rz*rz) > (sys->rcut+sqrt(2.0)*delta)) continue;
    //     //         if (sqrt(ry*ry+rz*rz) > (sys->rcut+sqrt(2.0)*delta)) continue;

    //     //         /* other cells that are too far apart */
    //     //         if (sqrt(rx*rx + ry*ry + rz*rz) > (sqrt(3.0)*delta+sys->rcut)) continue;
                
    //     //         /* cells are close enough. add to list */
    //     //         sys->plist[2*npair  ] = i;
    //     //         sys->plist[2*npair+1] = j;
    //     //         ++npair;
    //     //     }
    //     // }
    //     // sys->npair = npair;
    //     printf("Cell list has %dx%dx%d=%d cells with %d/%d pairs and "
    //            "%d atoms/celllist.\n", ngrid, ngrid, ngrid, sys->ncell, 
    //            sys->npair, ncell*(ncell-1)/2, nidx);
    // }

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

        // if (cell_index >= sys->ncell) {
        //     printf("Calculated cell index %d is out of bounds for atom %d\n", cell_index, i);
        //     cell_index = sys->ncell - 1;  // Clamp to max index
        // }
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

// Helper function to determine if an atom is near the boundary
int is_near_boundary(mdsys_t *sys, int i, double threshold) {
    double min_x = sys->rx[i] - sys->box;
    double max_x = sys->rx[i] + sys->box;
    double min_y = sys->ry[i] - sys->box;
    double max_y = sys->ry[i] + sys->box;
    double min_z = sys->rz[i] - sys->box;
    double max_z = sys->rz[i] + sys->box;

    // Check if the atom is within a threshold distance from any boundary
    return (min_x < threshold || max_x > sys->box - threshold ||
            min_y < threshold || max_y > sys->box - threshold ||
            min_z < threshold || max_z > sys->box - threshold);
}

// Setup MPI datatype for particle_data
void create_particle_data_type(MPI_Datatype *particle_type) {
    const int nitems = 2;
    int blocklengths[2] = {3, 3}; // There are 3 doubles for position and 3 doubles for velocity
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[2];

    offsets[0] = offsetof(particle_data, rx); // Offset of the first element
    offsets[1] = offsetof(particle_data, vx); // Offset of the first velocity element

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, particle_type);
    MPI_Type_commit(particle_type);
}


// // Function to identify boundary atoms
void identify_boundary_atoms(mdsys_t *sys, double threshold) {
    int i;
    for (i = 0; i < sys->natoms; i++) {
        if (is_near_boundary(sys, i, threshold)) {
            // Assuming sys->boundary_atoms is pre-allocated and enough space is available
            sys->boundary_atoms[sys->num_boundary_atoms++] = i;
        }
    }
}


/* Free cell list storage (including ghost lists) */
static void free_cell_list(mdsys_t *sys) {
    if (sys->clist != NULL) {
        for (int i = 0; i < sys->ncell; ++i) {
            free(sys->clist[i].idxlist);
            free(sys->clist[i].ghostlist);
      
        }
        free(sys->clist);
        sys->clist = NULL;
    }
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


double compute_interaction(mdsys_t *sys, int idx1, int idx2, double x1, double y1, double z1, double c12, double c6, double rcsq, double boxby2) {
    double dx = pbc(sys->rx[idx2] - x1, boxby2, sys->box);
    double dy = pbc(sys->ry[idx2] - y1, boxby2, sys->box);
    double dz = pbc(sys->rz[idx2] - z1, boxby2, sys->box);

    
    double rsq = dx * dx + dy * dy + dz * dz;
    double epot = 0.0;

    if (rsq < rcsq) {
        double rinv = 1.0 / rsq;
        double r6 = rinv * rinv * rinv;
        double ffac = (12.0 * c12 * r6 - 6.0 * c6) * r6 * rinv;
        sys->cx[idx1] += dx * ffac;
        sys->cy[idx1] += dy * ffac;
        sys->cz[idx1] += dz * ffac;
        sys->cx[idx2] -= dx * ffac;
        sys->cy[idx2] -= dy * ffac;
        sys->cz[idx2] -= dz * ffac;

        // Potential energy calculation
        epot = (c12 * r6 - c6) * r6;
    }
    return epot;
}

static void force(mdsys_t *sys) {
    double ffac, c12, c6, boxby2;
    double rx, ry, rz, rsq, rcsq;
    int i, j;
    int all_atoms = 0;
    // Precompute constants
    c12 = 4.0 * sys->epsilon * pow(sys->sigma, 12.0);
    c6  = 4.0 * sys->epsilon * pow(sys->sigma, 6.0);
    rcsq = sys->rcut * sys->rcut;
    boxby2 = 0.5 * sys->box;
    double epot = 0.0;

    azzero(sys->cx, sys->natoms);
    azzero(sys->cy, sys->natoms);
    azzero(sys->cz, sys->natoms);
    
    // // Zero out forces
    azzero(sys->fx, sys->natoms);
    azzero(sys->fy, sys->natoms);
    azzero(sys->fz, sys->natoms);

    MPI_Bcast(sys->rx, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    MPI_Bcast(sys->ry, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    MPI_Bcast(sys->rz, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    
    MPI_Bcast(sys->vx, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    MPI_Bcast(sys->vy, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    MPI_Bcast(sys->vz, sys->natoms, MPI_DOUBLE, 0, sys->mpicomm);
    
    // Calculate forces
    for (i = sys->start_cell; i < sys->end_cell; ++i) {
        all_atoms += sys->clist[i].natoms;
        printf("num of atoms in cell i: %d is %d, rank: %d\n", i, sys->clist[i].natoms, sys->mpirank);

        for (j = 0; j < sys->clist[i].natoms; ++j) {
            int idx1 = sys->clist[i].idxlist[j];
            double x1 = sys->rx[idx1], y1 = sys->ry[idx1], z1 = sys->rz[idx1];
            // Interaction with other real atoms
            for (int k = j + 1; k < sys->clist[i].natoms; ++k) {
                // printf("sys->clist[i].natoms: %d\n", sys->clist[i].natoms);
                int idx2 = sys->clist[i].idxlist[k];
                epot += compute_interaction(sys, idx1, idx2, x1, y1, z1, c12, c6, rcsq, boxby2);            
            }

            // Interaction with ghost atoms
            for (int k = 0; k < sys->clist[i].nghosts; ++k) {
                int idx2 = sys->clist[i].ghostlist[k];
                epot += compute_interaction(sys, idx1, idx2, x1, y1, z1, c12, c6, rcsq, boxby2);

            }
    
        }


    }


    printf("num of atoms: %d, rank: %d\n", all_atoms, sys->mpirank);

    // Sum local contributions to global arrays

    // Reduce forces across all MPI ranks
    MPI_Reduce(sys->cx, sys->fx, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    MPI_Reduce(sys->cy, sys->fy, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    MPI_Reduce(sys->cz, sys->fz, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    MPI_Reduce(&epot, &sys->epot, 1, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    

    // MPI_Reduce(sys->cx, sys->fx, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    // MPI_Reduce(sys->cy, sys->fy, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    // MPI_Reduce(sys->cz, sys->fz, sys->natoms, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);
    
    // MPI_Reduce(&epot, &sys->epot, 1, MPI_DOUBLE, MPI_SUM, 0, sys->mpicomm);

}
/* velocity verlet */
static void velverlet(mdsys_t *sys)
{
    int i;
    double dtmf;
    dtmf = 0.5*sys->dt / mvsq2e / sys->mass;

    MPI_Allreduce(MPI_IN_PLACE, sys->vx, sys->natoms, MPI_DOUBLE, MPI_SUM, sys->mpicomm);
    MPI_Allreduce(MPI_IN_PLACE, sys->vy, sys->natoms, MPI_DOUBLE, MPI_SUM, sys->mpicomm);
    MPI_Allreduce(MPI_IN_PLACE, sys->vz, sys->natoms, MPI_DOUBLE, MPI_SUM, sys->mpicomm);

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

// Helper function to determine if a cell index is local to the current MPI rank
int is_local(mdsys_t *sys, int cell_index) {
    return cell_index >= sys->start_cell && cell_index <= sys->end_cell;
}

int calculate_grid_dimensions(int nprocs) {
    return (int)ceil(sqrt(nprocs)); // Assuming a square grid for simplicity
}


void assign_cells_to_ranks(mdsys_t *sys) {
    int grid_dim = calculate_grid_dimensions(sys->nsize);
    int cells_per_row = sys->ngrid / grid_dim;
    
    int row = sys->mpirank / grid_dim;
    int col = sys->mpirank % grid_dim;

    sys->start_cell = (row * cells_per_row * sys->ngrid) + (col * cells_per_row);
    sys->end_cell = sys->start_cell + cells_per_row * sys->ngrid - 1;

    // Adjust for boundaries
    if (col == grid_dim - 1) { // Last column
        sys->end_cell += (sys->ngrid % grid_dim);
    }
    if (row == grid_dim - 1) { // Last row
        sys->end_cell += (sys->ngrid % grid_dim) * sys->ngrid;
    }

    printf("Rank: %d grid_dim: %d cells_per_row: %d start_cell: %d, end_cell %d \n",sys->mpirank, grid_dim, cells_per_row, sys->start_cell, sys->end_cell);
}

void find_neighbors(mdsys_t *sys, int cell_index, int *neighbors) {
    if (cell_index < 0 || cell_index >= sys->ncell) {
        fprintf(stderr, "Invalid cell index %d out of bounds [0, %d)\n", cell_index, sys->ncell);
        return;
    }

    int k, m, n;
    int ngrid = sys->ngrid;
    int dx, dy, dz;
    int nx, ny, nz;
    int count = 0;

    // Convert linear index to 3D grid coordinates
    k = cell_index / (ngrid * ngrid); // z-coordinate
    m = (cell_index % (ngrid * ngrid)) / ngrid; // y-coordinate
    n = cell_index % ngrid; // x-coordinate

    // Loop over all neighboring cells in 3x3x3 block
    for (dx = -1; dx <= 1; ++dx) {
        for (dy = -1; dy <= 1; ++dy) {
            for (dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue; // Skip the cell itself
                }

                // Periodic boundary conditions
                nx = (n + dx + ngrid) % ngrid;
                ny = (m + dy + ngrid) % ngrid;
                nz = (k + dz + ngrid) % ngrid;

                // Convert 3D coordinates back to linear index
                int neighbor_index = nz * ngrid * ngrid + ny * ngrid + nx;

                // Store the neighbor index
                if (count < 26) { // Ensure no overflow
                    neighbors[count++] = neighbor_index;
                }
            }
        }
    }

    // Ensure the list is properly terminated if not full
    if (count < 100) {
        neighbors[count] = -1;
    }
}


void update_ghost_lists(mdsys_t *sys) {
    int i, j;

    for (i = sys->start_cell; i < sys->end_cell; ++i) {
        if (sys->clist[i].ghostlist != NULL) {
            free(sys->clist[i].ghostlist); // Free the old ghost list
        }
        sys->clist[i].nghosts = 0; // Reset ghost count

        int *new_ghosts = NULL; // Start with a NULL pointer
        int nghosts = 0;

        int neighbors[100];  // Buffer for neighbor indices

        printf("Rank %d i: %d\n", sys->mpirank, i);
        find_neighbors(sys, i, neighbors);
        
        for (j = 0; neighbors[j] != -1; j++) {
            int neighbor_idx = neighbors[j];
            if (neighbor_idx < 0 || neighbor_idx >= sys->ncell) {
                fprintf(stderr, "Invalid neighbor index %d out of bounds [0, %d)\n", neighbor_idx, sys->ncell);
                continue;  // Skip invalid neighbor indices
            }
            if (!is_local(sys, neighbor_idx)) {
                cell_t *neighbor_cell = &sys->clist[neighbor_idx];
                for (int k = 0; k < neighbor_cell->natoms; k++) {
                    int atom_idx = neighbor_cell->idxlist[k];
                    if (is_near_boundary(sys, atom_idx, sys->rcut)) {
                        int *temp = realloc(new_ghosts, (nghosts + 1) * sizeof(int));
                        if (temp == NULL) {
                            perror("Failed to allocate memory for new_ghosts");
                            free(new_ghosts); // Clean up previously allocated memory
                            return; // Exit function on failure
                        }
                        new_ghosts = temp;
                        new_ghosts[nghosts++] = atom_idx;
                    }
                }
            }
        }
        sys->clist[i].ghostlist = new_ghosts;
        sys->clist[i].nghosts = nghosts;
    }
}

// Update to exchange_boundary_data function
void exchange_boundary_data(mdsys_t *sys, MPI_Datatype particle_data_type) {
    particle_data *sendbuf = malloc(sys->num_boundary_atoms * sizeof(particle_data));
    particle_data *recvbuf = malloc(sys->num_boundary_atoms * sizeof(particle_data));
    MPI_Status status;

    // Example neighbor setup, adjust according to your topology
    int neighbor = sys->mpirank + 1; // This is just an example, adjust as necessary

    MPI_Sendrecv(sendbuf, sys->num_boundary_atoms, particle_data_type, neighbor, 0,
                 recvbuf, sys->num_boundary_atoms, particle_data_type, neighbor, 0,
                 sys->mpicomm, &status);

    free(sendbuf);
    free(recvbuf);
}

void initialize_and_assign_cells(mdsys_t *sys) {
    int i, k, m, n, idx;
    double boxby2 = 0.5 * sys->box;
    double delta, boxoffs;

    // Calculate the grid dimensions and cell sizes
    sys->ngrid = floor(sys->box / (sys->rcut * cellrat));
    sys->ncell = sys->ngrid * sys->ngrid * sys->ngrid;
    delta = sys->box / sys->ngrid;
    boxoffs = boxby2 - 0.5 * delta;

    sys->delta = delta;

    // Allocate memory for cell list
    sys->clist = (cell_t *)malloc(sys->ncell * sizeof(cell_t));
    int nidx = 2 * sys->natoms / sys->ncell + 2; // Ensure enough space
    nidx = ((nidx / 2) + 1) * 2;  // Round up to the nearest even number
    sys->nidx = nidx;
    for (i = 0; i < sys->ncell; ++i) {
        sys->clist[i].idxlist = (int *)malloc(nidx * sizeof(int));
        sys->clist[i].natoms = 0;  // Initialize to zero
    }

    // Assign cells to ranks
    int grid_dim = calculate_grid_dimensions(sys->nsize);
    int cells_per_row = sys->ngrid / grid_dim;

    int row = sys->mpirank / grid_dim;
    int col = sys->mpirank % grid_dim;

    sys->start_cell = (row * cells_per_row * sys->ngrid) + (col * cells_per_row);
    sys->end_cell = sys->start_cell + cells_per_row * sys->ngrid - 1;

    // Adjust for boundaries in the last row or column
    if (col == grid_dim - 1) {
        sys->end_cell += (sys->ngrid % cells_per_row);
    }
    if (row == grid_dim - 1) {
        sys->end_cell += (sys->ngrid % cells_per_row) * sys->ngrid;
    }

    // Distribute atoms to cells
    for (i = 0; i < sys->natoms; ++i) {
        k = floor((pbc(sys->rx[i], boxby2, sys->box) + boxby2) / delta);
        m = floor((pbc(sys->ry[i], boxby2, sys->box) + boxby2) / delta);
        n = floor((pbc(sys->rz[i], boxby2, sys->box) + boxby2) / delta);
        int cell_index = n * sys->ngrid * sys->ngrid + m * sys->ngrid + k;

        if (cell_index >= sys->ncell) {
            printf("Calculated cell index %d is out of bounds for atom %d\n", cell_index, i);
            cell_index = sys->ncell - 1;  // Clamp to max index
        }

        idx = sys->clist[cell_index].natoms;
        sys->clist[cell_index].idxlist[idx] = i;
        ++(sys->clist[cell_index].natoms);
    }
        printf("Rank: %d grid_dim: %d cells_per_row: %d start_cell: %d, end_cell %d \n",sys->mpirank, grid_dim, cells_per_row, sys->start_cell, sys->end_cell);

}

/* main */
int main(int argc, char **argv) 
{
    // MPI Initialization phase
    MPI_Init( &argc, &argv);

    MPI_Datatype particle_data_type;
    create_particle_data_type(&particle_data_type);

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

        sys.ngrid = (int)ceil(sys.box / (sys.rcut * cellrat));
        sys.ncell = sys.ngrid * sys.ngrid * sys.ngrid;  // Total number of cells
        
    }
    // sys.nghosts = 10;
    MPI_Bcast(&sys.natoms, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.rcut, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.box, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.nsteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sys.dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&sys.ngrid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&sys.ncell, 1, MPI_INT, 0, MPI_COMM_WORLD);

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
            azzero(sys.fx, sys.natoms);
            azzero(sys.fy, sys.natoms);
            azzero(sys.fz, sys.natoms);
        } else {
            perror("cannot read restart file");
            return 3;
        }
    }

    MPI_Bcast(sys.rx, sys.natoms, MPI_DOUBLE, 0, sys.mpicomm);
    MPI_Bcast(sys.ry, sys.natoms, MPI_DOUBLE, 0, sys.mpicomm);
    MPI_Bcast(sys.rz, sys.natoms, MPI_DOUBLE, 0, sys.mpicomm);
    
    MPI_Bcast(sys.vx, sys.natoms, MPI_DOUBLE, 0, sys.mpicomm);
    MPI_Bcast(sys.vy, sys.natoms, MPI_DOUBLE, 0, sys.mpicomm);
    MPI_Bcast(sys.vz, sys.natoms, MPI_DOUBLE, 0, sys.mpicomm);

    if (sys.mpirank == 0) {
        erg=fopen(ergfile,"w");
        traj=fopen(trajfile,"w");
    }

    /* create initial cell list */
    sys.clist = NULL;
    sys.plist = NULL;
    // assign_cells_to_ranks(&sys);
    initialize_and_assign_cells(&sys);
    updcells(&sys);
    // update_ghost_lists(&sys);

    if (sys.mpirank == 0) {
        printf("ncell: %d\n", sys.ncell);
    }
    /* initialize forces and energies.*/
    sys.nfi=0;
    // force(&sys);
    // ekin(&sys);
    
    if (sys.mpirank == 0) {
        printf("Starting simulation with %d atoms for %d steps.\n",sys.natoms, sys.nsteps);
        printf("     NFI            TEMP            EKIN                 EPOT              ETOT\n");
        output(&sys, erg, traj);
    }
    /**************************************************/
    /* main MD loop */
    // for(sys.nfi=1; sys.nfi <= sys.nsteps; ++sys.nfi) {

    //     if (sys.mpirank == 0) {
    //     /* write output, if requested */
    //         if ((sys.nfi % nprint) == 0) {
    //             output(&sys, erg, traj);
    //         }
    // }
    //     /* propagate system and recompute energies */
    //     velverlet(&sys);
    //     ekin(&sys);

    //     /* update cell list */
    //     updcells(&sys);
    //     update_ghost_lists(&sys);

    // }
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
    if (sys.mpirank == 0)
    printf("Simulation Done.\n");
    MPI_Type_free(&particle_data_type);
    MPI_Finalize();

    return 0;
}
