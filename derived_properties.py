from lammps_io import LAMMPS_IO as LIO
import numpy as np

def rdf(
    io: LIO = None,
    species1: List = None,
    species2: List = None,
    rmin: int = 0,
    rmax: int = 8,
    nbins: int = 50,
    ) -> np.ndarray:

    ### Controls

    try:
        assert len(io.positions.shape)==3
    except AssertionError:
        print("Available positions should be a time trajectory")
        exit()

    if species1 is None:
        species1 = list(set(io.species))
        species2 = species1
    else:
        if species2 = None
            species2 = species1

    
    for t in range(io.positions.shape[0]):
        for i in range(io.positions.shape[1]-1):
            for j in range(i+1, io.positions.shape[1]):
                if io.species[i] in species1 and io.species[j] in species2:
                    rdf[]

    for s1 in species1:
        for s2 in species2:
            
          
    rdf = np.zeros(1)
    return rdf

def 
