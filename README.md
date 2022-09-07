# CASPT2 RDM Examples
In this repo you will find multidimensional expectation values (RDMs and CASPT2 intermediates) in the following formats:
 - `*.npy` are numpy arrays - they are read in and checked for mutual compatability in `checks.py` wherein their index ordering and algebraic convention is also specified
 - `spinfree*` are of the NECI-style plain text format, extended to higher body RDMs and intermediates. the indices are in normal order
 - `M7.caspt2.h5` is the M7-style HDF5 archive. Note that subgroups of `archive/rdms` now contain a dataset specifying whether the data are spinfree. Spinfree data can be assumed to be normalized since they are not suitable for restarts. The rows of the indices datasets are also in normal order

The system in question is the ground state of N2. and the methods used are RHF, CASSCF, and CASPT2. The converged CASSCF orbitals are used in the CASPT2 procedure, not the pseudo-canonical ones.

```
Basis set:               cc-pVDZ            
Density-fitting basis:   cc-pVDZ-jkfit      
Bond length:             1.098 Angstrom     
No. inactive MOs:        4                  
No. active MOs:          6                  
No. secondary MOs:       18                 
```

For more information about the calculation procedure, see the bagel input and output files.
