import numpy as np
from pde import PDEBase, FieldCollection, CartesianGrid, ScalarField,  MemoryStorage,plot_kymograph, plot_kymographs, solve_poisson_equation
from matplotlib import pyplot as plt

# PDENSTEch - Partial Differential Equation Numerical Solvert for Transient Electrochemistry

# A time-dependent solver for the 1D Poisson-Nernst-Planck Equations implemented using py-pde
# This is implemented to determine the transient concentration fields of an electrochemical system subject to time varying
#   applied fields. The user defines the initial chemical species and their concentrations, as well as the time varying
#   voltage profile.The simulator proceeds for a user-defined length of time and then plots the output time-dependent concentrations
#
# Structure of script:
#   Initialization parameter dictionary - a series of user defined parameters for setting up the simulation
#   PNP equation class - a set of functions which use the py-pde PDEBase class to define and run the PNP simulation
#       the Poisson equation is solved using py-pde.solve_poisson_equation. The applied voltage is used as a boundary condition
#       the NP equation is solved using the laplacian of the concentration (diffusion) and the gradient + divergence of the
#           electric field and component charge densities (drift)
#       a post-step hook function is used to change applied voltage (boundary condition) and evaluate metrics on the transient fields
#       the unpack function takes the initialization parameters it will need for the simulation and formats them appropriately
#   run_simulation is a wrapper which takes the user-supplied parameters and does the following:
#       convert user-supplied parameters to appropriate units (nm,ns,number)
#       create the initial concentration fields
#       initialize the PNP class and run the simulation
#       plot the results

params = {

}

class PNP(PDEBase):

    def __init__(self):
        """
        A PDE class for simulating the Poisson-Nernst-Planck (PNP) equations.

        Methods:
        unpack_parameters(params) -- takes a user-supplied dict of simulation parameters and converts the needed ones into
        into class variables. A list of needed keys are provided in the documentation of the unpack_parameters
        initialize_fields(params) -- uses the input parameters to generate the fields for the start of the simulation
        evolution_rate(state, t) -- the function that is run every iteration of the simulation in order to advance. This
        is not directly called by the user but is used by PDEBase.solve()
        post_step_hook(state_data, t) -- a function which is evaluated every iteration after evolution_rate. It is used in
        the PNP class to change the applied voltage and evaluate any user-defined metrics on the data
        solve(fields, t_range, dt, tracker) -- PDEBase method used to run the simulation. It is not modified here but is
        called by the run_simulation function elsewhere in the script
        plot_results(plotTypes, saveQ, saveName, saveDir) -- plots the results of the simulation
        """
        pass

    def unpack_parameters(self, params:dict):
        """
        Formats the input parameters for use in the simulation. Returns all values necessary to run PNP.solve()

        Args:
            params (dict): a dict of the simulation parameters
            params has several required keys. These are:

        Return:
            values necessary to run PNP.solve
            fields (FieldCollection): initial values of the concentration fields of each species in the simulation
            t_range (float): end time of the simulation
            dt (float): time step of the simulation
            tracker (list): specifications for the simulation tracker
        """
        pass

    def initialize_fields(self, params:dict):
        """
        Creates the initial concentration profile FieldCollection object for the start of the simulation

        Args:
            params (dict): a dict of simulation parameters
        Returns:
            FieldCollection: the initial concentration profiles of each simulated species. These are also saved as class variables
        """
        pass

    def evolution_rate(self, state, t=0):
        """
        The iteration function used at each step of the PNP simulation

        Args:
            state (FieldCollection): the current concentrations of each component of the system
            t (float): float, the current time of the simulation
        Returns:
            FieldCollection: change of state over time (i.e. dc/dt). Length must equal state

        Each iteration consists of three steps. First the charge density field is calculated from the current state. Next
        the electric potential field is found by solving the Poisson Equation using the charge density field. Finally, the
        change in concentration is evaluated using the concentration Laplacian (diffusion) and the divergence of the field
        gradient (drift)
        """
        pass

    def make_post_step_hook(self, state):
        """Create a hook function that is called after every time step"""

        def post_step_hook(self, state, t):
            """
            Update secondary parameters after every iteration of the simulation

            Args:
                state_data (FieldCollection): current state of the simulation
                t (float): simulation time
            Returns:
                hmmmm
            """
            pass

        return post_step_hook, 0.0 # hook function + initial data for t

    def plot_results(self, params:dict):
        """
        Plots the results of the simulation

        Args:
            params (dict): a dict of simulation parameters. The following keys are required to determine the function behavior:
            Keys:
            plotTypes (list): a list of strings for the type of plots to generate. If more than one type is listed, plots are
                generated one at a time in the order specified.
                Plot types: "kymographs", ...
            saveQ (boolean): should the plots be saved. All plots save at 300 dpi
            saveFormat (string): format the plots are saved in. Default is .pdf
            saveName (string): name of the plots that are saved. File names are saveName_plotType.saveFormat
            saveDir (string): directory to save the plots in
        """

        pass

def run_simulation(params:dict):
    """
    Wrapper function which runs the PNP simulation based on the supplied parameters

    params (dict): the simulation parameters supplied at the start of the script. The required keys are listed in the
        documentation PNP.unpack_parameters

    This function initializes the PNP class and calls the necessary pre-simulation methods, then runs the simulation, and
        plots the results
    """
    eq = PNP()
    solveParams = eq.unpack_parameters(params)
    result = eq.solve(*solveParams)
    eq.plot_results(params)
    return 0

run_simulation(params)