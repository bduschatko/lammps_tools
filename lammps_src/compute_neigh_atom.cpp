/* ----------------------------------------------------------------------
Blake R. Duschatko 

Compute a per-atom quantity giving the nearest neighbor to each atom 
------------------------------------------------------------------------- */

#include "compute_neigh_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeNeighAtom::ComputeNeighAtom(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), typelo(nullptr), typehi(nullptr), cvec(nullptr), carray(nullptr),
    group2(nullptr), normv(nullptr)
{
  if (narg < 5) error->all(FLERR, "Illegal compute neigh/atom command");

  jgroup = group->find("all");
  jgroupbit = group->bitmask[jgroup];
  cstyle = NONE;

  if (strcmp(arg[3], "cutoff") == 0) {
    cstyle = CUTOFF;
    double cutoff = utils::numeric(FLERR, arg[4], false, lmp);
    cutsq = cutoff * cutoff;

    int iarg = 5;
    if ((narg > 6) && (strcmp(arg[5], "group") == 0)) {
      group2 = utils::strdup(arg[6]);
      iarg += 2;
      jgroup = group->find(group2);
      if (jgroup == -1) error->all(FLERR, "Compute neigh/atom group2 ID does not exist");
      jgroupbit = group->bitmask[jgroup];
    }

  } else
    error->all(FLERR, "Invalid cstyle in compute neigh/atom");

  peratom_flag = 1;
  ncol = 2;
  size_peratom_cols = 2; // output atom ID and distance of nearest neighbor per atom

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeNeighAtom::~ComputeNeighAtom()
{
  if (copymode) return;

  delete[] group2;
  memory->destroy(cvec);
  memory->destroy(carray);
}

/* ---------------------------------------------------------------------- */

void ComputeNeighAtom::init()
{

  // TODO is it really needed to have a pair style? neigh list maybe wont exist

  if (force->pair == nullptr)
    error->all(FLERR, "Compute neigh/atom requires a pair style be defined");
  if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR, "Compute neigh/atom cutoff is longer than pairwise cutoff");

  // need an occasional full neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void ComputeNeighAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeNeighAtom::compute_peratom()
{
  int i, j, m, ii, jj, inum, jnum, jtype, n;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq, r;
  int *ilist, *jlist, *numneigh, **firstneigh;

  invoked_peratom = update->ntimestep;

  // grow compute array if necessary

  if (atom->nmax > nmax) {
    if (ncol == 1) {
      memory->destroy(cvec);
      nmax = atom->nmax;
      memory->create(cvec, nmax, "coord/atom:cvec");
      vector_atom = cvec;
    } else {
      memory->destroy(carray);
      nmax = atom->nmax;
      memory->create(carray, nmax, ncol, "coord/atom:carray");
      array_atom = carray;
    }
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute nearest neighbor distance for each atom in group

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    // initialize
    carray[i][0] = 0.0;
    carray[i][1] = 0.0;

    // apply to selected group only
    if (mask[i] & groupbit){

      // positions
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];

      // neighbors 
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (int jj = 0; jj < jnum; j++){

        // start at first neighbor
        j = jlist[jj];
        j &= NEIGHMASK; // mask out special bond indicator

        if (mask[j] & jgroupbit){

          delx = xtmp - x[j][0];
          dely = ytmp - y[j][0];
          delz = ztmp - z[j][0];

          rsq = delx * dely + dely * dely + delz * delz;
          r = sqrt(rsq);

          if (jj == 0){
            carray[i][0] = j;
            carray[i][1] = r;
          }
          elif (carray[i][1] > r) {
            carray[i][0] = j;
            carray[i][1] = r;
          }

        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeCoordAtom::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                        int * /*pbc*/)
{
  int i, m = 0, j;
  for (i = 0; i < n; ++i) {
    for (j = nqlist; j < nqlist + 2 * (2 * l + 1); ++j) { buf[m++] = normv[list[i]][j]; }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::unpack_forward_comm(int n, int first, double *buf)
{
  int i, last, m = 0, j;
  last = first + n;
  for (i = first; i < last; ++i) {
    for (j = nqlist; j < nqlist + 2 * (2 * l + 1); ++j) { normv[i][j] = buf[m++]; }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeCoordAtom::memory_usage()
{
  double bytes = (double) ncol * nmax * sizeof(double);
  return bytes;
}
