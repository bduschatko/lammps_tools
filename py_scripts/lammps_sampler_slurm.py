from flare_cg.utils.lammps.lammps_sampler import LAMMPS_Sampler
from flare_cg.utils.mpi import is_variable_allowed, convert_var_string, \
                                    verify_allocated_resources, \
                                    variable_not_found_error


class LAMMPS_SLURM(LAMMPS_Sampler):
    """
    This class creates an infrastructure for the use of the LAMMPS MD engine 
    to perform dynamics. Using a SLURM scheduler, the class can run N different
    processes with the same parameters. This is useful to parallelizing the
    constrained dynamics process in CG applications. 

    Args:
        
        nProcesses (int): The number of parallel processes to run
        executable_path (str): File path to the LAMMPS executable to use
        input_template (str): File path to an input file to be used as the 
                                reference template for creating new inputs
        slurm_vars (dict): A dictionary of slurm variables to be used when 
                            running parallel processes
        lammps_vars (dict): A dictionary of user defined LAMMPS variables,
                            giving the name (key) and value

    For additional parameters, see 
                        flare_cg.base_classes.sampler

    """

    def __init__(self,
                slurm_vars: dict={}, 
                **kwargs
                ) -> None:

        super().__init__(**kwargs)

        verify_allocated_resources(self.nProcesses, slurm_vars)
        self.slurm_vars = slurm_vars


    def _init_properties(self) -> None:
        super()._init_properties()
        self.slurm_vars = {}
        return None


    def _create_execution_type(self, pid: int):

        process_command = "srun "

        # loop through slurm vars
        for var in self.slurm_vars.keys():
            if is_variable_allowed(var):
                process_command += "--{}={} ".format(convert_var_string(var), 
                                                        self.slurm_vars[var])
            else:
                raise KeyError(variable_not_found_error)

        # set process directory
        pdir = self.process_path_list[pid]
        process_command += "--chdir={} ".format(pdir)
        process_command += f"--exact " #--exclusive " # --relative={pid} "
        
        # set output
        process_command += "--output={}/{}.out ".format(pdir, self.base_name.format(pid))

        return process_command


    # Properties ---------------------------------------------------------------


    @property
    def nProcesses(self):
        return self._nProcesses

    @nProcesses.setter
    def nProcesses(self, value):
        self._nProcesses = value

    @property
    def slurm_vars(self):
        return self._slurm_vars

    @slurm_vars.setter
    def slurm_vars(self, value):
        self._slurm_vars = value
