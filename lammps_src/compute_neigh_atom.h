/* -*- c++ -*- ----------------------------------------------------------
Blake R. Duschatko 

Compute a per-atom quantity giving the nearest neighbor to each atom 
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(neigh/atom,ComputeNeighAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_NEIGH_ATOM_H
#define LMP_COMPUTE_NEIGH_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeNeighAtom : public Compute {
 public:
  ComputeNeighAtom(class LAMMPS *, int, char **);
  ~ComputeNeighAtom() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double memory_usage() override;
  enum { NONE, CUTOFF, ORIENT };

 protected:
  int nmax, ncol;
  double cutsq;
  class NeighList *list;

  double *cvec; // compute scalar per atom
  double **carray; // compute vector per atom

  char *group2;
  int jgroup, jgroupbit;

  double threshold;
  double **normv;
  int cstyle; // style of the compute, only supports CUTOFF
  int nqlist, l;
};

}    // namespace LAMMPS_NS

#endif
#endif
