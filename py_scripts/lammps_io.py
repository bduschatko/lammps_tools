import numpy as np
from os.path import abspath
from typing import List
from time import time_ns

from flare_cg.base_classes.io import IO

"""
Global Variables ---------------------------------------------------------------

    sections (dict): A dictionary mapping the LAMMPS data file section names
                     to the corresponding attribute name used in the IO class

    headers (dict): A dictionary mapping the LAMMPS data file headers to the 
                    corresponding attribute name used in the IO class

    scalar_attributes (dict): A mapping from attribute names used in LAMMPS 
                              files to the attribute names in the IO class for 
                              variables that are scalar values for each atom

    vector_attributes (dict): A mapping from a tuple of attribute names used in 
                              LAMMPS files to the attribute names in the IO 
                              class for variables that are vector values for 
                              each atom 

"""

# Data File Elements -----------------------------------------------------------

sections = {
        "Masses" : "masses_by_type",
        "Pair Coeffs" : "pair_coeffs",
        "Bond Coeffs" : "bond_coeffs",
        "Angle Coeffs" : "angle_coeffs",
        "Dihedral Coeffs" : "dihedral_coeffs",
        "Improper Coeffs" : "improper_coeffs",
        "Atoms" : None,
        "Velocities" : "velocities",
        "Bonds" : "bonds",
        "Angles" : "angles",
        "Dihedrals" : "dihedrals",
        "Impropers" : "impropers"
}


headers = {
    "atoms" : "nAtoms",
    "atom types" : "nAtomTypes",
    "bonds" : "nBonds",
    "bond types" : "nBondTypes",
    "angles" : "nAngles",
    "angle types" : "nAngleTypes",
    "dihedrals" : "nDihedrals",
    "dihedral types" : "nDihedralTypes",
    "impropers" : "nImpropers",
    "improper types" : "nImproperTypes"
}

# Dump File Elements -----------------------------------------------------------

scalar_attributes = {
    "id" : "ids",
    "mol" : "molecules", 
    "type" : "species",
    "mass" : "masses",
    "mu" : "dipoles",  
    "q" : "charges",
}

vector_attributes = {
    ("x", "y", "z") : "positions",
    ("xs", "ys", "zs") : "scaled_positions",
    ("nx", "ny", "nz") : "image_flag",
    ("vx", "vy", "vz") : "velocities",
    ("fx", "fy", "fz") : "forces",
    ("mux", "muy", "muz") : "dipole_vectors", 
    ("omegax", "omegay", "omegaz") : "angular_velocities",
    ("angmomx", "angmomy", "angmomz") : "angular_momentum",
    ("tqx", "tqy", "tqz") : "torques"
}

# Parser Class -----------------------------------------------------------------

class LAMMPS_IO(IO):
    """
    This class interfaces with LAMMPS input and output files to read and write
    similar ones. Parsed files have all relevant attributes saved to class 
    variables. When writing new files, the existing attributes that have been
    read previously can be used so that only portions of the files need 
    replacing. 
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


    # Publicly accessed functions ----------------------------------------------


    def parse(self, file_path: str=None, style: str="full",
                reset: bool=False, sort: bool=False) -> None: 
        """
        Parse a LAMMPS data or dump file

        Args

            file_path (str): location of the file to read
            style (str): type of lammps file to read (e.g. dump, full, etc)
            reset (bool): Set whether or not any existing properties in the IO
                          class should be reset. If not, all properties remain
                          unchanged if they are not read from the new file
            sort (bool): Set whether or not to sort parsed properties according
                         to their atom ID

        """

        # reset parsed properties
        super().parse(reset)

        if style=="full":
            self._lammps_data(file_path, style=style, reset=reset, sort=sort)
        elif style=="dump":
            self._lammps_dump(file_path, reset=reset, sort=sort)
        else:
            raise NotImplementedError(
                    "File type {} not supported".format(style))

        return None


    def write(self, save_to: str=None,
                    style: str="full",
                    step_number: int=-1,
                    replacements: dict={},
                    id_list: List=None) -> str:
        """
        Write a LAMMPS data file given a subset of entries 

        : step_number : is the time stamp ID for which to write information.
                        if available data is for a single step, this is written.
                        Defaults to -1 (most recent step)

        : replacements : a dictionary of LAMMPS sections to overwrite in the IO
                        class

        : id_list : the ID ordering of the provided property replacements, used
                    to ensure the replacements are ordered the same way as 
                    stored properties


        CAUTION : for per-atom properties, previously parsed properties may be 
                  ordered differently from the new replacement properties. ALL 
                  PER ATOM REPLACEMENTS MUST GET SORTED BY ID LIST TO MATCH
                  THE EXISTING IDS ORDER

        """

        # Save current attributes that will be replaced 
        previous_save = {}

        # OPTIONAL replace stored properties with given ones
        for key in replacements.keys():

            # sort if this is a per-atom property
            # self id's is gauranteed to be defined and gives current order
            if len(replacements[key]) == self.nAtoms and id_list is not None:

                args = [np.where(id_list == i)[0][0] for i in self.ids]
                replacements[key] = replacements[key][args]

            elif len(replacements[key]) == self.nAtoms and id_list is None:
                raise AttributeError("Per-atom properties being replaced must "
                                        "have an id_list to indicate how they "
                                        "are ordered.")

            if key in scalar_attributes.keys():
                previous_save[scalar_attributes[key]] = getattr(
                                                self, scalar_attributes[key])
                setattr(self, scalar_attributes[key], replacements[key])
            elif key in scalar_attributes.values():
                previous_save[key] = getattr(self, key)
                setattr(self, key, replacements[key])
            elif key in vector_attributes.keys():
                previous_save[vector_attributes[key]] = getattr(
                                                self, vector_attributes[key])
                setattr(self, vector_attributes[key], replacements[key])
            elif key in vector_attributes.values():
                previous_save[key] = getattr(self, key)
                setattr(self, key, replacements[key])
            else:
                raise Warning("IO class has no attribute" 
                            " {} to write".format(key))

        # write header 
        contents = self._write_header()

        for num, key in enumerate(sections.keys()):

            # REQUIRED write atoms section for structure file
            if key == "Atoms":
                contents += self._write_lammps_atoms(style)
            else:
                # OPTIONAL write any stored section attributes that are not None
                att = getattr(self, sections[key])
                if att is not None:
                    
                    contents += key + "\n\n"

                    if key=="Masses":
                        for i in att.keys():
                            contents += "{} {}\n".format(i, str(att[i]))

                    elif key=="Improper Coeffs":
                        for i, line in enumerate(att):
                            contents += "{} ".format(i+1)
                            for item in range(len(line)):
                                if item==0:
                                    contents += "{} ".format(line[item])
                                else:
                                    contents += "{} ".format(int(line[item]))
                            contents += "\n"         

                    # if time series take last element
                    else:
                        try:
                            if len(att.shape)==3:
                                att = att[step_number]
                        except AttributeError:
                            if len(np.array(att).shape)==3:
                                att = att[step_number]

                        if key=="Velocities":

                            for i, line in enumerate(att):
                                contents += "{} ".format(self.ids[i])
                                for item in line:
                                    contents += "{} ".format(item)
                                contents += "\n"

                        else:

                            for i, line in enumerate(att):
                                contents += "{} ".format(i+1)
                                for item in line:
                                    contents += "{} ".format(item)
                                contents += "\n"

                    if num != len(sections.keys())-1: contents += "\n"

        # remove trailing white space
        if contents[-1] == "\n":
            contents = contents[:-1]

        # write unique file name to save to
        if save_to is None:
            save_to = "{}.data".format(time_ns())

        with open(save_to, "w") as f:
            f.write(contents)

        # Reset replaced attributes to previous values
        for key in previous_save:
            setattr(self, key, previous_save[key])

        return abspath(save_to)


    def get_property_dictionary(self, subset: List=None) -> dict:
        """
        Return a dictionary of properties stored in the IO class.

        Args:

            subset (list): All attributes to get back from the IO class. If none
                            are given, all properties in a full type data file
                            will be returned
        """

        if subset is None:
            subset = [
                "positions",
                "velocities",
                "forces",
                "energies",
                "stresses",
                "masses",
                "charges",
                "cell",
                "bonds",
                "angles",
                "dihedrals"
                "pair_coeffs",
                "bond_coeffs",
                "angle_coeffs",
                "dihedral_coeffs"
            ]

        return super().get_property_dictionary(subset)


    # Privately accessed functions ---------------------------------------------


    def _init_properties(self) -> None:
        super()._init_properties()

        self.nAtomTypes = None
        self.nBondTypes = None
        self.nAngleTypes = None
        self.nDihedralTypes = None
        self.nImproperTypes = None

        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.impropers = None

        self.charges = None

        #self.image_flags = None

        self.bond_topology = None
        self.atoms_in_molecule = None
        self.masses_by_type = None

        self.xlo = None
        self.xhi = None
        self.ylo = None
        self.yhi = None
        self.zlo = None
        self.zhi = None
        self.xy = None
        self.xz = None
        self.yz = None
        self.cell = None

        for key in scalar_attributes.keys():
            setattr(self, scalar_attributes[key], None)

        for key in vector_attributes.keys():
            setattr(self, vector_attributes[key], None)

        return None


    def _reset_dump_properties(self) -> None:
        """
        Reset IO attributes associated with LAMMPS dump files that may have been
        read previously
        """

        for key in scalar_attributes.keys():
            setattr(self, scalar_attributes[key], None)

        for key in vector_attributes.keys():
            setattr(self, vector_attributes[key], None)

        self.cell = None
        #self.image_flags = None

        return None


    def _reset_data_properties(self) -> None:
        """
        Reset IO attributes associated with LAMMPS data files that may have been
        read previously
        """

        for key in sections.keys():
            if key != "Atoms":  
                setattr(self, sections[key], None)

        for key in headers.keys():
            setattr(self, headers[key], None)

        self.ids = None
        self.positions = None
        self.velocities = None
        self.charges = None
        self.species = None
        self.molecules = None
        self.atoms_in_molecule = None
        self.bond_topology = None
        self.cell = None
        #self.image_flags = None

        self.xlo = None
        self.xhi = None
        self.ylo = None
        self.yhi = None
        self.zlo = None
        self.zhi = None
        self.xy = None
        self.xz = None
        self.yz = None

        return None


    def _sort_dump_properties(self) -> None:
        """
        Sort the data parsed from a dump file. These are given as time series
        data, so sorting must take this into consideration 
        """

        indices = np.argsort(self.ids, axis=1)

        for key in scalar_attributes.keys():
            att = getattr(self, scalar_attributes[key])
            if att is not None:
                setattr(self, scalar_attributes[key], 
                        np.take_along_axis(
                            att, indices, axis=1))

        for key in vector_attributes.keys():
            att = getattr(self, vector_attributes[key])
            if att is not None:
                setattr(self, vector_attributes[key], 
                    np.take_along_axis(att,indices[:,:,None],axis=1))

        return None


    def _sort_data_properties(self) -> None:
        """
        Sort atomic attributes parsed from a data file according to atom ID
        """

        scalars = ["masses", "charges", "molecules", "species", "ids"]
        vectors = ["positions", "velocities"]

        indices = np.argsort(self.ids, axis=0)

        for name in scalars:
            att = getattr(self, name)
            if att is not None:
                setattr(self, name, 
                        np.take_along_axis(
                            att, indices, axis=0))
        
        for name in vectors:
            att = getattr(self, name)
            if att is not None:
                setattr(self, name, 
                        np.take_along_axis(att,indices[:,None],axis=0))

        for key in self.atoms_in_molecule:
            self.atoms_in_molecule[key] = sorted(self.atoms_in_molecule[key])

        return None


    def _set_dump_properties(self, data: np.ndarray, 
                                    items: List,
                                    sort: bool=True) -> None:
        """
        Given a numpy array of data parsed from a dump file with identifiers 
        given by items, set IO attributes accordingly 
        """

        # loop through attributes and find elements to avoid repeats

        for key in scalar_attributes.keys():
            if key in items:
                index = items.index(key)
                setattr(self, scalar_attributes[key], data[:,:,index])

        for key in vector_attributes.keys():
            try:
                indices = [items.index(key[0]), 
                            items.index(key[1]), 
                                items.index(key[2])]

                setattr(self, vector_attributes[key], 
                                data[:,:,indices])
            except:
                pass

        # sort
        if sort and self.ids is None:
            raise ValueError("Sorting cannot be done without atom IDs")
        elif sort:
            self._sort_dump_properties()

        return None


    def _set_topology(self) -> None:
        """
        Create a bond topology object given the bond information for a system. 
        The topology is a dictionary of lists. The key is an atom ID, and the
        value is a sorted List of atom ID's that are bonded to that atom
        """

        self.bond_topology = {}

        for i in self.ids:

            # locate the unique list of atoms this is bonded to
            indices_0 = np.where(self.bonds[:,1] == i)[0]
            indices_1 = np.where(self.bonds[:,2] == i)[0]

            set0 = self.bonds[indices_0,2]
            set1 = self.bonds[indices_1,1]

            if len(set0)==0 or len(set1)==0:

                if len(set0)==0:
                    bonded_atoms = np.unique(set1)
                else:
                    bonded_atoms = np.unique(set0)                    

            else:
                bonded_atoms = np.unique(np.concatenate((set0, set1)))

            self.bond_topology[i] = np.array(sorted(bonded_atoms))

        return None


    # section writers ----------------------------------------------------------

    def _write_header(self) -> str:
        """
        Write the header information for a LAMMPS data file
        """

        header = "LAMMPS data file via flare_cg.utils.LAMMPS_IO\n\n"
        
        lines_added = False
        for key in headers.keys():
            # OPTIONAL write any stored header information
            att = getattr(self, headers[key])
            if att is not None:
                header += "{} {}\n".format(str(att), key)
                lines_added = True

        if lines_added: header += "\n"

        if self.cell is not None:
            cell = self._write_cell()
            header += cell

        return header


    def _write_cell(self) -> str:
        """
        Write the cell of a system to a LAMMPS data file. See LAMMPS
        documentation for definitions on cell parameters 
        """

        cell_string = ""

        # rectangular box
        cell_string += "0.0 {} xlo xhi\n"\
                        "0.0 {} ylo yhi\n"\
                        "0.0 {} zlo zhi\n".format(
                            self.cell[0,0], self.cell[1,1], self.cell[2,2]
                        )

        if self.cell[1,0] != 0 or self.cell[2,0] !=0 or self.cell[2,1] != 0:
            cell_string +="{} {} {} xy xz yz\n".format(
                        self.cell[1,0], self.cell[2,0], self.cell[2,1])

        cell_string += "\n"

        return cell_string


    def _write_lammps_atoms(self, style: str) -> str:
        if style=="full": 
            return self._write_lammps_atoms_full()
        else:
            return "\n"


    def _write_lammps_atoms_full(self, ids: np.ndarray = None) -> str:
        """
        Write the Atoms section of a LAMMPS data file in the full format
        """

        if self.positions is not None:
            try:
                atoms = "Atoms # full \n\n"
                for i in range(self.nAtoms):
                    pos = self.positions
                    pos = pos.reshape((-1, pos.shape[-2], pos.shape[-1]))[-1]
                    line = "{} {} {} {} {} {} {}".format(
                            self.ids[i],
                            self.molecules[i],
                            self.species[i],
                            self.charges[i],
                            pos[i, 0], pos[i, 1], pos[i, 2]
                    )
                    if pos.shape[1] > 3: # image flags
                        line += " {} {} {}".format(int(pos[i, 3]), 
                                                    int(pos[i, 4]), 
                                                    int(pos[i, 5]))
                    atoms += line + "\n"

                atoms += "\n"
            except:
                if self.positions is None:
                    pass
                else:
                    raise RuntimeError("An error occured while writing the "
                                    "atoms section of structure file. All "
                                    "properties of the full style may not "
                                    "have been stored in the IO class object.")

            return atoms

        else:
            return ""


    # section parsers ----------------------------------------------------------

    # data file sections ----------------------------------

    def _parse_lammps_atoms(self, data: List, style: str) -> None:
        if style=="full": 
            self._parse_lammps_atoms_full(data)
        
        return None


    def _parse_lammps_atoms_full(self, data: List) -> None:
        """
        Parse the Atoms section of a LAMMPS data file having the full style
        """

        types = []
        charges = []
        atoms_in_molecule = {}
        positions = []
        molecules = []
        ids = []
        masses = []
        for line in data:
            tokens = line.strip().split()
            ids.append(int(tokens[0]))
            molecules.append(int(tokens[1]))
            types.append(int(tokens[2]))
            masses.append(self.masses_by_type[int(tokens[2])])
            charges.append(float(tokens[3]))
            if len(tokens) == 10: # image flags
                positions.append(np.array([
                            float(tokens[4]),
                            float(tokens[5]),
                            float(tokens[6]),
                            float(tokens[7]),
                            float(tokens[8]),
                            float(tokens[9])]
                ))
            else:
                positions.append(np.array([
                            float(tokens[4]),
                            float(tokens[5]),
                            float(tokens[6])]
                ))
            if int(tokens[1]) not in atoms_in_molecule.keys():
                atoms_in_molecule.update({int(tokens[1]) : []})
            atoms_in_molecule[int(tokens[1])].append(int(tokens[0]))

        self.ids = np.array(ids).astype(int)
        self.positions = np.array(positions)
        if self.positions.shape[1] > 3:
            self.image_flags = self.positions[:,-3:]
        self.positions = self.positions[:,:3]
        self.charges = np.array(charges)
        self.species = np.array(types).astype(int)
        self.molecules = np.array(molecules)
        self.masses = np.array(masses)
        self.atoms_in_molecule = atoms_in_molecule

        return None


    def _parse_lammps_masses(self, data: List) -> None:
        """
        Parse the mass section of a LAMMPS data file. This creates a dictionary
        mapping atom type to atom mass
        """

        masses = {}
        for line in data:
            tokens = line.strip().split()
            masses[int(tokens[0])] = float(tokens[1])

        self.masses_by_type = masses

        return None


    def _parse_lammps_pair_coeffs(self, data: List) -> None:
        """
        Parse the pair coefficients from a LAMMPS data file. Each entry is a 
        unique type, and the values are the LJ parameters
        """

        pair_coeffs = []
        for line in data:
            tokens = line.strip().split()
            pair_coeffs.append(np.array([float(tokens[1]),
                                        float(tokens[2])]))

        self.pair_coeffs = np.array(pair_coeffs)

        return None


    def _parse_lammps_bonds(self, data: List) -> None:
        """
        Parse the bonds section from a LAMMPS data file. This indicates the bond
        type, as well as the two atoms involved in the bond
        """

        bonds = []
        for line in data:
            tokens = line.strip().split()
            bonds.append(np.array([int(tokens[1]),
                                    int(tokens[2]), 
                                    int(tokens[3])]))

        self.bonds = np.array(bonds).astype(int)

        return None


    def _parse_lammps_bond_coeffs(self, data: List) -> None:
        """
        Parse the bond coefficients from a LAMMPS data file. Each entry is a 
        unique type, and the values are the spring constant and displacement
        """
        
        bond_coeffs = []
        for line in data:
            tokens = line.strip().split()
            bond_coeffs.append(np.array([float(tokens[1]), 
                                        float(tokens[2])]))

        self.bond_coeffs = np.array(bond_coeffs)

        return None


    def _parse_lammps_angles(self, data: List) -> None:
        """
        Parse the angles section from a LAMMPS data file. This indicates the 
        angle type, as well as the three atoms involved in the angle
        """

        angles = []
        for line in data:
            tokens = line.strip().split()
            angles.append(np.array([int(tokens[1]),
                                    int(tokens[2]), 
                                    int(tokens[3]),
                                    int(tokens[4])]))

        self.angles = np.array(angles).astype(int)

        return None


    def _parse_lammps_angle_coeffs(self, data: List) -> None:
        """
        Parse the angle coefficients from a LAMMPS data file. Each entry is a 
        unique type, and the values are the spring constant and displacement
        """

        angle_coeffs = []
        for line in data:
            tokens = line.strip().split()
            angle_coeffs.append(np.array([float(tokens[1]), 
                                        float(tokens[2])]))

        self.angle_coeffs = np.array(angle_coeffs)

        return None


    def _parse_lammps_dihedrals(self, data: List) -> None:
        """
        Parse the dihedrals section from a LAMMPS data file. This indicates the 
        dihedral type, as well as the four atoms involved in the dihedral
        """

        dihedrals = []
        for line in data:
            tokens = line.strip().split()
            dihedrals.append(np.array([int(tokens[1]),
                                    int(tokens[2]), 
                                    int(tokens[3]),
                                    int(tokens[4]),
                                    int(tokens[5])]))

        self.dihedrals = np.array(dihedrals).astype(int)

        return None
        

    def _parse_lammps_dihedral_coeffs(self, data: List) -> None:
        """
        Parse the dihedral coefficients from a LAMMPS data file. Each entry is a 
        unique type, and the values are the terms in an OPLS style dihedral 
        """

        dihedral_coeffs = []
        for line in data:
            tokens = line.strip().split()

            dihedral_coeffs.append([])
            for t in tokens[1:]:
                if t=="#":
                    break
                elif t.lstrip("-").isdigit():
                    dihedral_coeffs[-1].append(int(t))
                else:
                    dihedral_coeffs[-1].append(float(t))

        self.dihedral_coeffs = dihedral_coeffs #np.array(dihedral_coeffs)

        return None


    def _parse_lammps_impropers(self, data: List):

        impropers = []
        for line in data:
            tokens = line.strip().split()
            impropers.append(np.array([int(tokens[1]),
                                    int(tokens[2]), 
                                    int(tokens[3]),
                                    int(tokens[4]),
                                    int(tokens[5])]))

        self.impropers = np.array(impropers).astype(int)

        return None
        


    def _parse_lammps_improper_coeffs(self, data: List):

        improper_coeffs = []
        for line in data:
            tokens = line.strip().split()
            improper_coeffs.append([])
            for t in tokens[1:]:
                if t.lstrip("-").isdigit():
                    improper_coeffs[-1].append(int(t))
                else:
                    improper_coeffs[-1].append(float(t))

        self.improper_coeffs = improper_coeffs

        return None

    def _parse_lammps_velocities(self, data: List) -> None:
        """
        Parse the velocities section of a LAMMPS data file
        """

        velocities = []
        for line in data:
            tokens = line.strip().split()
            velocities.append(np.array([float(tokens[1]),
                                        float(tokens[2]),
                                        float(tokens[3])]))

        self.velocities = np.array(velocities)

        return None


    # dump file sections ----------------------------------


    def _parse_box(self, data: List) -> np.ndarray:
        """
        Parse the system box information from a LAMMPS dump file and convert 
        this information into a cell object. See the LAMMPS documentation for
        details on box to cell definitions
        """

        # orthogonal box
        if len(data[0].strip().split()) == 2:

            xTokens = data[0].strip().split()
            yTokens = data[1].strip().split()
            zTokens = data[2].strip().split()

            local_cell = np.eye(3)
            local_cell[0,0] = float(xTokens[1]) - float(xTokens[0])
            local_cell[1,1] = float(yTokens[1]) - float(yTokens[0])
            local_cell[2,2] = float(zTokens[1]) - float(zTokens[0])

        # non orthogonal box, see LAMMPS documentation
        else:

            l1Tokens = data[0].strip().split()
            l2Tokens = data[1].strip().split()
            l3Tokens = data[2].strip().split()

            local_cell = np.eye(3)
            local_cell[0] = np.array([
                        float(l1Tokens[1])-float(l1Tokens[0]), 0, 0])
            local_cell[1] = np.array([
                        float(l1Tokens[2]), 
                        float(l2Tokens[1]) - float(l2Tokens[0]), 0])
            local_cell[2] = np.array([
                        float(l2Tokens[2]), float(l3Tokens[2]), 
                        float(l3Tokens[1]) - float(l3Tokens[0])
            ])

        return local_cell


    # File Readers -------------------------------------------------------------

    def _lammps_data(self, data_file: str, 
                            style: str, 
                            reset: bool,
                            sort: bool) -> None:
        """
        Parse a LAMMPS data file

        Args:

            data_file (str): file to parse
            style (str): style of the file, e.g. full
            reset (bool): whether to reset previously parsed properties
            sort (bool): whether to sort atom properties by ID
        """

        if reset: self._reset_data_properties()

        openFile = open(data_file,"r")
        data = openFile.readlines()
        openFile.close()

        nLines = len(data)
        line_count = 0
        done = False

        is_header = False
        is_section = False

        while not done:

            line = data[line_count].strip()

            if len(line)==0:
                line_count += 1
                continue

            # skip comments
            elif line[0] == "#":
                line_count += 1
                continue

            for key in headers.keys():
                if key in line:
                    is_header = True
                    header_key = key
                    break

            if not is_header:
                for section in sections.keys():
                    if section in line:
                        is_section = True
                        section_key = section
                        break

            if is_header:

                if header_key == "atoms": 
                    self.nAtoms = int(line.split()[0])
                if header_key == "atom types": 
                    self.nAtomTypes = int(line.split()[0])
                if header_key == "bonds": 
                    self.nBonds = int(line.split()[0])
                if header_key == "bond types": 
                    self.nBondTypes = int(line.split()[0])
                if header_key == "angles": 
                    self.nAngles = int(line.split()[0])
                if header_key == "angle types": 
                    self.nAngleTypes = int(line.split()[0])
                if header_key == "dihedrals": 
                    self.nDihedrals = int(line.split()[0])
                if header_key == "dihedral types": 
                    self.nDihedralTypes = int(line.split()[0])
                if header_key == "impropers": 
                    self.nImpropers = int(line.split()[0])
                if header_key == "improper types": 
                    self.nImproperTypes = int(line.split()[0])

                line_count += 1
                is_header = False

            elif is_section:

                if section_key == "Masses": 
                    self._parse_lammps_masses(
                        data[line_count+2 : line_count+2+self.nAtomTypes]
                    )
                    line_count += self.nAtomTypes + 2

                elif section_key == "Bonds":
                    self._parse_lammps_bonds(
                        data[line_count+2 : line_count+2+self.nBonds]
                    )
                    line_count += self.nBonds + 2

                elif section_key == "Angles":
                    self._parse_lammps_angles(
                        data[line_count+2 : line_count+2+self.nAngles]
                    )
                    line_count += self.nAngles + 2

                elif section_key == "Dihedrals":
                    self._parse_lammps_dihedrals(
                        data[line_count+2 : line_count+2+self.nDihedrals]
                    )
                    line_count += self.nDihedrals + 2

                elif section_key == "Impropers":
                    self._parse_lammps_impropers(
                        data[line_count+2 : line_count+2+self.nImpropers]
                    )
                    line_count += self.nImpropers + 2

                elif section_key == "Pair Coeffs":
                    self._parse_lammps_pair_coeffs(
                        data[line_count+2 : line_count+2+self.nAtomTypes]
                    )
                    line_count += self.nAtomTypes + 2

                elif section_key == "Bond Coeffs":
                    self._parse_lammps_bond_coeffs(
                        data[line_count+2 : line_count+2+self.nBondTypes]
                    )
                    line_count += self.nBondTypes + 2

                elif section_key == "Angle Coeffs":
                    self._parse_lammps_angle_coeffs(
        
                        data[line_count+2 : line_count+2+self.nAngleTypes]
                    )
                    line_count += self.nAngleTypes + 2

                elif section_key == "Dihedral Coeffs":
                    self._parse_lammps_dihedral_coeffs(
                        data[line_count+2 : line_count+2+self.nDihedralTypes]
                    )
                    line_count += self.nDihedralTypes + 2

                elif section_key == "Improper Coeffs":
                    self._parse_lammps_improper_coeffs(
                        data[line_count+2 : line_count+2+self.nImproperTypes]
                    )
                    line_count += self.nImproperTypes + 2

                elif section_key == "Velocities":
                    self._parse_lammps_velocities(
                        data[line_count+2 : line_count+2+self.nAtoms]
                    )
                    line_count += self.nAtoms + 2

                elif section_key == "Atoms":
                    self._parse_lammps_atoms(
                        data[line_count+2 : line_count+2+self.nAtoms], style
                    )
                    line_count += self.nAtoms + 2

                else: print(section_key)

                is_section = False

            # construct cell
            elif "xlo xhi" in line:
                tokens = line.split()
                self.xlo = float(tokens[0])
                self.xhi = float(tokens[1])

                tokens = data[line_count + 1].strip().split()
                self.ylo = float(tokens[0])
                self.yhi = float(tokens[1])

                tokens = data[line_count + 2].strip().split()
                self.zlo = float(tokens[0])
                self.zhi = float(tokens[1])
                
                if "xy xz yz" in data[line_count + 3].strip():
                    tokens = data[line_count + 3].strip().split()
                    self.xy = float(tokens[0])
                    self.xz = float(tokens[1])
                    self.yz = float(tokens[2])
                    self.cell = np.array([
                                    [self.xhi-self.xlo, 0, 0],
                                    [self.xy, self.yhi-self.ylo, 0],
                                    [self.xz, self.yz, self.zhi-self.zlo]
                    ])
                    line_count += 4

                else:
                    self.cell = np.diag([self.xhi-self.xlo, 
                                            self.yhi-self.ylo,
                                            self.zhi-self.zlo])
                    line_count += 3

            else:
                line_count += 1

            # end of file
            if line_count >= nLines: 
                done = True

        if sort: self._sort_data_properties()
        self._set_topology()

        return None


    def _lammps_dump(self, dump_file: str, reset: bool, sort: bool) -> None:
        """
        Parse a LAMMPS dump file 

        Args:

            dump_file (str): file to parse
            reset (bool): whether to reset previously parsed properties
            sort (bool): whether to sort atom properties by ID 
        """

        if reset: self._reset_dump_properties()

        openFile = open(dump_file,"r")
        data = openFile.readlines()
        openFile.close()

        nLines = len(data)
        line_count = 0
        
        # first header
        nAtoms = int(data[3].strip())
        items = data[8].strip().split()[2:]

        timesteps = int(nLines / (9+nAtoms))

        contents = np.zeros((timesteps, nAtoms, len(items)))
        cells = np.zeros((timesteps, 3, 3))

        for t in range(timesteps):

            cells[t] = self._parse_box(data[line_count+5 : line_count+8])
            line_count += 9
            contents[t] = np.loadtxt(
                            " ".join(data[line_count:line_count+nAtoms]
                            ).strip().splitlines())
            line_count += nAtoms

        self.cell = cells
        self._set_dump_properties(contents, items, sort)
        
        return None


    # Parsed Properties --------------------------------------------------------

    @property 
    def nAtomTypes(self):
        return self._nAtomTypes

    @nAtomTypes.setter
    def nAtomTypes(self, value):
        self._nAtomTypes = value

    @property 
    def nBondTypes(self):
        return self._nBondTypes

    @nBondTypes.setter
    def nBondTypes(self, value):
        self._nBondTypes = value

    @property 
    def nAngleTypes(self):
        return self._nAngleTypes

    @nAngleTypes.setter
    def nAngleTypes(self, value):
        self._nAngleTypes = value

    @property 
    def nDihedralTypes(self):
        return self._nDihedralTypes

    @nDihedralTypes.setter
    def nDihedralTypes(self, value):
        self._nDihedralTypes = value

    @property 
    def nImproperTypes(self):
        return self._nImproperTypes

    @nImproperTypes.setter
    def nImproperTypes(self, value):
        self._nImproperTypes = value

    @property
    def bonds(self):
        return self._bonds

    @bonds.setter
    def bonds(self, value):
        self._bonds = value

    @property
    def angles(self):
        return self._angles 

    @angles.setter
    def angles(self, value):
        self._angles = value

    @property
    def dihedrals(self):
        return self._dihedrals 

    @dihedrals.setter
    def dihedrals(self, value):
        self._dihedrals = value 

    @property
    def impropers(self):
        return self._impropers 

    @impropers.setter
    def impropers(self, value):
        self._impropers = value

    @property
    def charges(self):
        return self._charges

    @charges.setter
    def charges(self, value):
        self._charges = value

    @property
    def molecules(self):
        return self._molecules 
    
    @molecules.setter
    def molecules(self, value):
        self._molecules = value

    @property 
    def masses_by_type(self):
        return self._masses_by_type 

    @masses_by_type.setter
    def masses_by_type(self, value):
        self._masses_by_type = value

    @property
    def xlo(self):
        return self._xlo

    @xlo.setter
    def xlo(self, value):
        self._xlo = value

    @property
    def xhi(self):
        return self._xhi

    @xhi.setter
    def xhi(self, value):
        self._xhi = value

    @property
    def ylo(self):
        return self._ylo

    @ylo.setter
    def ylo(self, value):
        self._ylo = value

    @property
    def yhi(self):
        return self._yhi

    @yhi.setter
    def yhi(self, value):
        self._yhi = value

    @property
    def zlo(self):
        return self._zlo

    @zlo.setter
    def zlo(self, value):
        self._zlo = value

    @property
    def zhi(self):
        return self._zhi

    @zhi.setter
    def zhi(self, value):
        self._zhi = value

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, value):
        self._xy = value

    @property
    def xz(self):
        return self._xz

    @xz.setter
    def xz(self, value):
        self._xz = value

    @property
    def yz(self):
        return self._yz

    @yz.setter
    def yz(self, value):
        self._yz = value

    # Dump file properties -----------------------------------------------------

    @property
    def ids(self):
        return self._ids 
    
    @ids.setter
    def ids(self, value):
        self._ids = value

    @property 
    def unwrapped_positions(self):
        return self._unwrapped_positions 

    @unwrapped_positions.setter
    def unwrapped_positions(self, value):
        self._unwrapped_positions = value

    @property
    def scaled_unwrapped_positions(self):
        return self._scaled_unwrapped_positions

    @scaled_unwrapped_positions.setter 
    def scaled_unwrapped_positions(self, value):
        self._scaled_unwrapped_positions = value

    @property
    def dipole_vectors(self):
        return self._dipole_vectors 
    
    @dipole_vectors.setter
    def dipole_vectors(self, value):
        self._dipole_vectors = value

    @property
    def dipoles(self):
        return self._dipoles 
    
    @dipoles.setter
    def dipoles(self, value):
        self._dipoles = value
        
    @property
    def angular_velocities(self):
        return self._angular_velocities

    @angular_velocities.setter
    def angular_velocities(self, value):
        self._angular_velocities = value

    @property
    def angular_momentum(self):
        return self._angular_momentum 

    @angular_momentum.setter
    def angular_momentum(self, value):
        self._angular_momentum = value

    @property
    def torques(self):
        return self._torques

    @torques.setter
    def torques(self, value):
        self._torques = value

    # Derived Properties -------------------------------------------------------

    @property 
    def atoms_in_molecule(self):
        return self._atoms_in_molecule 
    
    @atoms_in_molecule.setter
    def atoms_in_molecule(self, value):
        self._atoms_in_molecule = value

    @property
    def bond_topology(self):
        return self._bond_topology

    @bond_topology.setter
    def bond_topology(self, value):
        self._bond_topology = value
