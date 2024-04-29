
## Compile

```bash
gcc -o ljmd.x ljmd.c -lm
```

## install

```bash
./ljmd.x < argon_108.inp 
```


## input sample

```
108               # natoms
39.948            # mass in AMU
0.2379            # epsilon in kcal/mol
3.405             # sigma in angstrom
8.5               # rcut in angstrom
17.1580           # box length (in angstrom)
argon_108.rest    # restart
argon_108.xyz     # trajectory
argon_108.dat     # energies
10000             # nr MD steps
5.0               # MD time step (in fs)
100               # output print frequency
```