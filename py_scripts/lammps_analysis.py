import numpy as np
from numpy.linalg import norm
from flare_cg.utils.lammps.lammps_io import LAMMPS_IO
from flare_cg.utils.geometry import periodic_boundary_vector_wrap as pbvw

def compute_rdf(parser: LAMMPS_IO,
                bins = 10,
                rrange = (0,8),
                subspecies=[]):

    if len(subspecies)==0:
        positions = parser.positions
    else:
        positions = parser.positions[:,subspecies]

    natoms = positions.shape[1]

    box = np.zeros((3,2))
    box[:,1] = np.diag(parser.cell[0])

    edges = np.linspace(rrange[0], rrange[1], bins+1)
    rdf = np.zeros(len(edges)-1)

    for t in range(len(positions)):
        print(t)
        #[norm(pbvw(positions[t,i]-positions[t,j],box)) for i in range(natoms-1) for j in range(i+1, natoms)]
        for i in range(natoms-1):
            for j in range(i+1, natoms):
                r = norm(pbvw(positions[t,i]-positions[t,j], box))
                # locate bin
                if r < edges[-1]:
                    loc = np.where(r < edges)[0][0]
                    rdf[loc-1] += 1

    rdf = rdf/np.sum(rdf)
    for r in range(rdf.shape[0]):
        rdf[r] = rdf[r]/4/np.pi/(edges[r] + (edges[1]-edges[0])/2)**2
    return rdf

