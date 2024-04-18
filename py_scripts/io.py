import numpy as np
from typing import List

class IO():
    """
    A class used for managing all-atom information originating from a file. The
    parent class assumes no file format, but does provide proprety definitions 
    that may appear in some all-atom files. 
    """

    def __init__(self, **kwargs) -> None:
        self._init_properties()
        return None


    def _init_properties(self) -> None:
        """
        Create all properties of the class, set to None
        """

        self.nAtoms = None
        self.nBonds = None
        self.nAngles = None
        self.nDihedrals = None
        self.nImpropers = None

        self.pair_coeffs = None
        self.bond_coeffs = None
        self.angle_coeffs = None
        self.dihedral_coeffs = None
        self.improper_coeffs = None
        self.dihedral_coeffs = None

        self.cell = None
        self.masses = None
        self.velocities = None
        self.positions = None
        self.scaled_positions = None
        self.forces = None
        self.energies = None
        self.stresses = None

        self.ids = None
        self.species = None
        self.properties = None

        return None


    def parse(self, reset: bool=True, **kwargs) -> None:
        """
        Read a single files contents 

        : reset : whether or not to set all previously recorded values to None
        """

        # reset parsed properties
        if reset:
            self._init_properties()

        return None


    def write(self) -> None:
        """
        write contents to a file
        """
        return None


    def get_property_dictionary(self, subset: List=[]) -> dict:
        """
        Group a set of parsed attributes into a dictionary

        : subset : a list of properties to get back from the parser.
                   If none, no properties will be given
        """

        properties = {}

        if len(subset) > 0:
            for var in subset:
                try:
                    properties[var] = getattr(self, var)
                except:
                    raise KeyError(
                                "Attribute {} could not be found".format(var))

        return properties
        

    # Properties --------------------------------------------------------------

    @property 
    def nAtoms(self):
        return self._nAtoms

    @nAtoms.setter
    def nAtoms(self, value):
        self._nAtoms = value

    @property
    def ids(self):
        return self._ids 
    
    @ids.setter
    def ids(self, value):
        self._ids = value

    @property
    def pair_coeffs(self):
        return self._pair_coeffs

    @pair_coeffs.setter
    def pair_coeffs(self, value):
        self._pair_coeffs = value

    @property
    def nBonds(self):
        return self._nBonds

    @nBonds.setter
    def nBonds(self, value):
        self._nBonds = value

    @property
    def bond_coeffs(self):
        return self._bond_coeffs

    @bond_coeffs.setter
    def bond_coeffs(self, value):
        self._bond_coeffs = value

    @property
    def nAngles(self):
        return self._nAngles

    @nAngles.setter
    def nAngles(self, value):
        self._nAngles = value
        
    @property
    def angle_coeffs(self):
        return self._angle_coeffs

    @angle_coeffs.setter
    def angle_coeffs(self, value):
        self._angle_coeffs = value

    @property
    def nDihedrals(self):
        return self._nDihedrals

    @nDihedrals.setter
    def nDihedrals(self, value):
        self._nDihedrals = value
        
    @property
    def dihedral_coeffs(self):
        return self._dihedral_coeffs

    @dihedral_coeffs.setter
    def dihedral_coeffs(self, value):
        self._dihedral_coeffs = value

    @property
    def nImpropers(self):
        return self._nImpropers

    @nImpropers.setter
    def nImpropers(self, value):
        self._nImpropers = value
        
    @property
    def improper_coeffs(self):
        return self._improper_coeffs

    @improper_coeffs.setter
    def improper_coeffs(self, value):
        self._improper_coeffs = value

    @property
    def masses(self):
        return self._masses

    @masses.setter
    def masses(self, value):
        self._masses = value

    @property 
    def species(self):
        return self._species 

    @species.setter
    def species(self, value):
        self._species = value
    
    @property
    def velocities(self):
        return self._velocities

    @velocities.setter
    def velocities(self, value):
        self._velocities = value

    @property 
    def positions(self):
        return self._positions 

    @positions.setter
    def positions(self,value):
        self._positions = value

    @property
    def scaled_positions(self):
        return self._scaled_positions 

    @scaled_positions.setter
    def scaled_positions(self, value):
        self._scaled_positions = value

    @property
    def forces(self):
        return self._forces 

    @forces.setter
    def forces(self, value):
        self._forces = value

    @property
    def energies(self):
        return self._energies 

    @energies.setter
    def energies(self, value):
        self._energies = value

    @property
    def stresses(self):
        return self._stresses

    @stresses.setter
    def stresses(self, value):
        self._stresses = value 

    @property
    def cell(self):
        return self._cell 
    
    @cell.setter
    def cell(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("Cell(s) must be a numpy array")
        self._cell = value

    @property 
    def properties(self):
        return self._properties 
    
    @properties.setter
    def properties(self, value):
        self._properties = value