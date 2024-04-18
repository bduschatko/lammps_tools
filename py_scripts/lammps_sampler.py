from os.path import join, exists

from shutil import which, copyfile

from flare_cg.base_classes.sampler import Sampler
from flare_cg.utils.mpi import is_mpi_variable_allowed, \
                                    convert_var_string, \
                                    variable_not_found_error, \
                                    is_mpi_variable_allowed
from flare_cg.utils.lammps.lammps_io import LAMMPS_IO


class LAMMPS_Sampler(Sampler):
    """
    This class creates an infrastructure for the use of the LAMMPS MD engine 
    to perform dynamics.  

    Args:
        
        executable_path (str): File path to the LAMMPS executable to use
        input_template (str): File path to an input file to be used as the 
                                reference template for creating new inputs
        lammps_vars (dict): A dictionary of user defined LAMMPS variables,
                            giving the name (key) and value

    For additional parameters, see 
                        flare_cg.base_classes.sampler

    """

    def __init__(self,
                nProcesses: int=1,
                executable_path: str=None,
                input_template: str=None,
                structure_template: str=None,
                mpi_vars: dict={},
                lammps_vars: dict={},
                **kwargs
                ) -> None:

        super().__init__(**kwargs)

        self.nProcesses = nProcesses

        self.IO = LAMMPS_IO()

        # set up all atom information from a template, used to generate new ones
        if structure_template is not None:
            self.IO.parse(file_path=structure_template, 
                    style = "full",
                    sort = True)


        if executable_path is None:

            try:
                if which("lmp_mpi") is None:
                    self.executable_path = which("lmp")
                else:
                    self.executable_path = which("lmp_mpi")

                if self.executable_path is None:
                    raise FileNotFoundError(
                                "A LAMMPS executable could not be found")
            except:
                raise RuntimeError("An issue occured with locating LAMMPS")

        else:

            if exists(executable_path):
                self.executable_path = executable_path
            else:
                raise RuntimeError("An issue occured with locating LAMMPS")

        # name templates of data and input files to be created
        self.base_name = "process_{}"
        self.input_template = input_template
        self.structure_template = structure_template

        self.lammps_vars = lammps_vars
        self.mpi_vars = mpi_vars
        self._init_private_lammps_vars()


    def _init_properties(self) -> None:
        super()._init_properties()

        self.executable_path = None
        self.input_template = None

        self.base_name = None

        self.mpi_vars = {}
        self.lammps_vars = {}
        self.private_lammps_vars = {}

        return None


    def _init_private_lammps_vars(self) -> None:
        """
        These optional private variables can be entered into the input template
        and will be replaced by the below variable definitions. This is 
        helpful for streamlining the processes of creating many input files. 

        Alternatively, different names can be entered in the input template
        explicitly and will not be overwritten. 
        """

        self.private_lammps_vars = {
            "_STRUCTURE" : "process_{}.data",
            "_PRODUCTION_DUMP" : "process_{}.dump",
            "_PRODUCTION_LOG" : "process_{}.log",
            "_PRODUCTION_DATA" : "process_{}.equilibrated-data" 
        }

        return None


    # create input files for process id
    def _generate_inputs(self, system, pid: int, template: str = None) -> None:
        """
        For a given process with identification number id, create an input file
        using the provided template or the one stored in the sampler at creation

        The input is saved in the working directory of this process
        """

        # create input file by copying input template
        copyfile(self.input_template, 
                join(self.process_path_list[pid], self.base_name.format(pid) + ".in"))

        # use initial structure template
        if template is None:
            template = self.structure_template

        self.IO.parse(file_path=template, 
                        style = "full",
                        reset = True, 
                        sort = True)

        # structure file
        replacements = {
            "positions" : system.positions
        }
        save_file = join(self.process_path_list[pid], self.base_name.format(pid) + ".data")
        self.IO.write(style="full", 
                        replacements=replacements,
                        save_to=save_file,
                        id_list=system.arrays["ids"])


        return None


    def _create_execution_type(self, pid: int) -> str:

        process_command = "mpirun "

        # loop through MPI vars
        for var in self.mpi_vars.keys():
            if is_mpi_variable_allowed(var):
                process_command += "-{} {} ".format(convert_var_string(var), 
                                                        self.mpi_vars[var])
            else:
                raise KeyError(variable_not_found_error.format(var))


        # set process directory
        pdir = self.process_path_list[pid]
        process_command += "-wdir {} ".format(pdir)
        
        process_command += "-merge-stderr-to-stdout --mca orte_base_help_aggregate 0 "
        #process_command += "--output-filename {}/{}.out ".format(pdir, self.base_name.format(pid))

        return process_command 


    def _update_command_string(self, command_string: str, pid: int) -> str:
        """
        Update the command to be run using mpi syntax. Each process is 
        appended to the existing command string to be run simultaneously with
        the other processes. 
        """

        process_command = self._create_execution_type(pid)

        # set executbale
        process_command += self.executable_path + " "

        # add private vars
        for var in self.private_lammps_vars.keys():
            process_command += \
                        "-var {} {} ".format(var, 
                        self.private_lammps_vars[var].format(pid))

        # add custom vars 
        for var in self.lammps_vars.keys():
            process_command += "-var {} {} ".format(var, self.lammps_vars[var])

        # set input file
        process_command += "-in " + self.base_name.format(pid) + ".in "

        # set log file
        #process_command += "-log " + self.base_name.format(pid) + ".log " 

        # set output file
        #process_command += "> " + join(self.process_path_list[pid], self.base_name.format(pid)) + ".out "

        process_command += "& "
        command_string = command_string +  process_command

        return command_string


    # Properties ---------------------------------------------------------------


    @property
    def input_name(self):
        return self._input_name

    @input_name.setter
    def input_name(self, value):
        self._input_name = value

    @property
    def output_name(self):
        return self._output_name

    @output_name.setter
    def output_name(self, value):
        self._output_name = value

    @property
    def mpi_vars(self):
        return self._mpi_vars 

    @mpi_vars.setter
    def mpi_vars(self, value):
        self._mpi_vars = value

    @property 
    def lammps_vars(self):
        return self._lammps_vars

    @lammps_vars.setter
    def lammps_vars(self, value):
        self._lammps_vars = value

    @property
    def private_lammps_vars(self):
        return self._private_lammps_vars 
    
    @private_lammps_vars.setter
    def private_lammps_vars(self, value):
        self._private_lammps_vars = value
