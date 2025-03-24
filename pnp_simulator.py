import numpy as np
from pde import PDEBase, FieldCollection, CartesianGrid, ScalarField,  MemoryStorage,plot_kymograph, plot_kymographs, solve_poisson_equation
from matplotlib import pyplot as plt
import math
import voltage_functions as vf
import pickle
import os
import time

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
    #todo: move parameters that are defined in terms of other parameters to unpack_params method
    #todo: convert species dependent parameters to index-matched lists
    #todo: handle unit conversions in unpack params
    # define constants. Units are m, eV, e, K, V

    ####################################################
    ####### Chemical Components ####################
    ##################################################

    # note: all lists describing the chemical species to simulate MUST have equal lengths
    "names": ["Mg", "Cl"],  # list of strings: names of each species in simulation
    "charges": [2, -1],  # list of integers: charges of each species
    "diffusivities": [2.2 * (10 ** -5), 2.2 * (10 ** -5)], # list of floats: diffusion constant of each species, in cm^2/s
    "bulkConcentrations": [0.05, 0.1],  # list of floats: concentration at far edge of simulation box, in mol/L
    "initialConcentrationProfiles": ["bulk", "bulk"],  # list of floats or ndarrays: initial concentration profiles
                                                        # options:
                                                        # "bulk" to match the bulkConcentration of the species
                                                        # "zero" to set initial to zero
                                                        # array: set custom values at each point.
                                                        #   must be a numpy array. If the length is longer than the number of grid points,
                                                        #   extra values will be cut off. If the array is shorter than the number of grid points,
                                                        #   an IndexError will be raised
    "solventDielectric" : 1, # float: relative dielectric constant of the solvent
    "temperature": 300,  # float: temperature in K. Raising temperature increases the ratio of diffusion to drift

    ##################################################
    ######### Simulation Conditions ##################
    ################################################

    "voltageFunction" : vf.cosVoltage, # function: must take the state and time as arguments and
                            # output a single float to be used as the applied voltage.
    "voltageFunctionArgs" : [0.0005, 0.1, 0.05], # list: extra arguments to pass to the voltage function, if needed
                                # if no extra arguments are needed, use an empty list
    "gridMax" : 1, # float: distance from the electrode surface to simulate, in nm
    "gridStep": 0.01,  # float: distance between grid points, in nm
    "tStep": 0.00001,  # float: time step interval for each simulation iteration, in ns
    "tStop": 0.005,  # float: time the simulation will run to. Number of steps = tStop/tStep
    "trackerStep" : 0.00005, # float: time step interval that the tracker will record data, in ns
                           # this should always be greater than tStep and less than tStop
                           # selecting a smaller number will give more information in plots but at the cost of memory usage
    # NOTE ON NUMERICAL STABILITY: stability of the simulation is sensitive to the ratio of gridStep to tStep
    #   this is because derivative boundary conditions will cause the results to be divided by the gridStep, resulting in larger values
    #   at the same time, diffusion can only spread large changes in concentration out by one grid point each iteration
    #   as a result, large changes at the edges from the gridStep need to be moderated by a smaller tStep
    #   empirically, a 10x decrease in gridStep requires a 100x decrease in tStep to maintain stability.
    #   0.01 gridStep and 0.00001 tStep is a good starting point

    ################################################
    ###### Physical Constants #####################
    ################################################

    "vacuumPermitivitty" : 0.055263, # e^2/eVnm
    "charge" : 1, # electron charge (elem charge)
    "kb" : 8.617*(10**-5), # boltzmann constant (eV/K)
    "Av" : 6.022 * (10**23), # Avogadro's number

    ##############################################
    ####### Plotting and Saving #################
    ############################################
    "plotQ" : True, # boolean: should a plots be generated after running the simulation?
    "saveQ" : False, # boolean: should the result of the simulation and plots (if made) be pickled (saved) when the simulation finishes?
    "showPlotsQ": True,  # boolean: should the generated plots be shown after they are generated?
    "saveName" : "default", # string: the base name of the simulation to save associated date. File names are saveName_dataType.saveFormat
    "saveDir" : "", # string: the directory to save data in
    "plotTypes" : ["concVsTime"], # list of strings: what types of plots should be generated.
                        # Current options are "kymographs", "concVsTime",
    "plotArgs" : {"concVsTime":[0,5,20]}, # dict: key is a plotType and val is a list of extra parameters to pass into the plotting function
                     #      Currently used for "concVsTime" to specify what points to plot
    "plotSaveFormat" : ".pdf" # string: format to save the plots
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
        plot_data(plotTypes, saveQ, saveName, saveDir) -- plots the results of the simulation
        check_index_matching(lists, keys) -- checks that a list of lists all have the same length, raises an exception otherwise
        """
        pass

    def unpack_parameters(self, params:dict):
        """
        Formats the input parameters for use in the simulation. Returns all values necessary to run PNP.solve()

        Args:
            params (dict): a dict of the simulation parameters
            params has several required keys. These are:
                "names", "charges", "diffusivities", "bulkConcentrations", "initialConcentrationProfiles", "solventDielectric",
                "temperature", "voltageProfile", "gridMax", "gridStep", "tStep", "tStop", "trackerStep", "vacuumPermitivitty",
                "charge", "kb", "Av", "plotQ", "saveQ", "saveName", "saveDir", "plotTypes", "plotSaveFormat"
        Return:
            values necessary to run PNP.solve
            fields (FieldCollection): initial values of the concentration fields of each species in the simulation
            t_range (float): end time of the simulation
            dt (float): time step of the simulation
            tracker (list): specifications for the simulation tracker

        Several class variables are initialized, including constants that are used in every simulation iteration as well
        as the first set of boundary conditions:
            self.c_bcs: boundary conditions for each concentration field. Used to calculate diffusion derivatives
            self.ePot_bc: boundary conditions for the electric potential. Used to solve Poisson's Equation and calculate the potential gradient
                Note that this boundary condition can change over the course of the simulation as specified by the voltage profile
            self.drift_bc: boundary conditions for calculating the drift contribution to dc/dt (divergence of the concentration * potential gradient)
        """
        # check that parameters are properly index matched
        # for now the index matched keys are just hard coded here. This may change in the future
        indexMatchedKeys = ["names", "charges", "diffusivities", "bulkConcentrations", "initialConcentrationProfiles"]
        indexMatchedLists = [params[key] for key in indexMatchedKeys]
        self.check_index_matching(indexMatchedLists, indexMatchedKeys)

        # save the initial parameters for later reference
        self.params = params
        self.dt = params["tStep"]

        # calculate values that will be used in every simulation iteration and save them as class variables
        # lists are converted to arrays in order to multiply by floats
        self.dielectric = params["solventDielectric"] * params["vacuumPermitivitty"] # absolute solvent dielectric constant in e^2/eVnm
        self.diffusivities = (10**5) * np.array(params["diffusivities"]) # convert diffusivities from cm^2/s to nm^2/ns
        self.kt = params["kb"] * params["temperature"] # boltzmann constant * temperature in eV
        self.bulk_concs = (params["Av"] / (10**24)) * np.array(params["bulkConcentrations"]) # bulk concentrations converted from mol/L to number / nm^3
        self.zs = np.array(params["charges"])
        self.numberOfSpecies = len(self.zs) # number of species comes up a lot later

        # calculate the debye length, just because it's good to know and we might want it later
        # print a note comparing the debye length to the grid size. grid size should be >> debye length
        self.bulkChargeDensity = sum([(self.zs[i]**2) * self.bulk_concs[i] for i in range(self.numberOfSpecies)])
        self.debye = np.sqrt((self.dielectric * self.kt)/self.bulkChargeDensity)
        print("Grid size is " + str(params["gridMax"]) + " nm")
        print("Debye length is " + str(self.debye) + " nm")

        # initialize grid and concentration fields
        self.initConcFields = self.initialize_fields(params)

        # gather the voltage function and calculate the first voltage at t=0
        # this must be done after fields are initialized
        self.voltageFunction = params["voltageFunction"]
        self.voltageArgs = params["voltageFunctionArgs"]
        self.initVoltage = self.voltageFunction(0, self.initConcFields, *self.voltageArgs)

        # determine boundary conditions and save them as class variables
        # todo: check that the bc explanations are correct and consistent
        # concentration boundary conditions for calculating diffusion:
        #   Left derivative is zero since no flux comes from the electrode
        #   Right value (far edge away from electrode) is the bulk value
        self.c_bcs = [{"x-": {"derivative": 0}, "x+": {"value": conc}} for conc in self.bulk_concs]
        # initial electric potential boundary conditions for solving the Poisson equation and later calculating the gradient of the potential
        #   Left derivative is set to the applied voltage since the derivative of potential is field
        #   Right value is set to 0 since there should be not electric potential at the far end of the simulation
        self.ePot_bc = {"x-": {"derivative": self.initVoltage}, "x+": {"value": 0}}
        # drift boundary conditions for determining the concentration change due to electromigration
        #   Left value is 0 since no concentration change comes from the electrode
        #   Right derivative is 0 since no flux due to drift can occur from the far boundary
        self.drift_bc = {"x-": {"value": 0}, "x+": {"derivative": 0}}

        # create tracker
        self.storage = MemoryStorage()
        trackerSpecification = ["progress", self.storage.tracker(params["trackerStep"])]

        # return inputs to PNP.solve()
        return self.initConcFields, params["tStop"], params["tStep"], trackerSpecification

    def initialize_fields(self, params:dict):
        """
        Creates the initial concentration profile FieldCollection object for the start of the simulation

        Args:
            params (dict): a dict of simulation parameters
        Returns:
            FieldCollection: the initial concentration profiles of each simulated species. These are also saved as class variables
        """
        # initialize the grid
        numberOfPoints = math.floor(params["gridMax"] / params["gridStep"]) # math.floor used to ensure numberOfPoints is an integer
        grid = CartesianGrid([(0, params["gridMax"])], [numberOfPoints], False)

        # for each species, create a field using the corresponding init profile
        concArrays = []
        for i in range(self.numberOfSpecies):

            initProfile = params["initialConcentrationProfiles"][i]

            # if "bulk" match bulk conc
            if initProfile == "bulk":
                profileArray = np.ones(numberOfPoints) * self.bulk_concs[i]

            # if "zero" set to 0
            elif initProfile == "zero":
                profileArray = np.zeros(numberOfPoints)

            # if a custom array, first check that it is the correct length (raise error if not), than create
            elif type(initProfile) == np.ndarray:

                # handle malformed inputs
                if len(initProfile) < numberOfPoints:
                    IndexError("PNP.initialize_fields: a supplied initial concentration profile for species " +
                               str(params["names"][i])) + (" is shorter than the grid length specified by gridMax and "
                                                           "gridStep. Simulation cannot run until this is addressed.")
                elif len(initProfile) > numberOfPoints:
                    print("PNP.initialize_fields: a supplied initial concentration profile for species " +
                               str(params["names"][i])) + (" is longer than the grid length specified by gridMax and "
                                                           "gridStep. Excess values beyond the grid length will be ignored.")

                profileArray = initProfile[:numberOfPoints] # slice cuts off excess values. Warning about this situation is raised above

            concArrays.append(profileArray)

        # gather all concArrays into a FieldCollection and return
        return FieldCollection([ScalarField(grid, concArray) for concArray in concArrays])

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

        concs = state  # state is the next set of concentration fields

        # generate charge density field from c by summing all charges times concentration
        chargeConcs = [self.zs[i] * concs[i] for i in range(self.numberOfSpecies)]
        p = sum(chargeConcs)

        # define Poisson Equation right hand side (-p/dielectric)
        poissonRHS = -1 * p / self.dielectric

        # solve for electric potential by Poisson's equation
        ePotential = solve_poisson_equation(poissonRHS, self.ePot_bc)

        # determine the electric field by taking the gradient of the potential
        eField = ePotential.gradient(bc=self.ePot_bc)

        # solve for the drift and diffusion components of the Nernst-Planck equations, then combine for total dc/dt
        drift = [((self.diffusivities[i] * chargeConcs[i] / self.kt) * eField).divergence(bc = self.drift_bc) for i in range(self.numberOfSpecies)]
        diff = [self.diffusivities[i] * concs[i].laplace(bc = self.c_bcs[i]) for i in range(self.numberOfSpecies)]
        dcdt = [drift[i] + diff[i] for i in range(self.numberOfSpecies)]

        # calculate the voltage for the next step
        newVoltage = self.voltageFunction(t + self.dt, state, *self.voltageArgs)
        self.ePot_bc = {"x-": {"derivative": newVoltage}, "x+": {"value": 0}}

        return FieldCollection(dcdt)

    #todo: update documentation about abandoning post step hook. the necessary operations can be updated within the
    #       evolution function and the post_step_hook is complicated :(

    # def make_post_step_hook(self, state):
    #     """Create a hook function that is called after every time step"""
    #
    #     def post_step_hook(self, state, t):
    #         """
    #         Update the voltage after every iteration of the simulation using self.voltageFunction
    #
    #         Args:
    #             state_data (FieldCollection): current state of the simulation
    #             t (float): simulation time
    #         Returns:
    #             None
    #         """
    #         #calculate new voltage, update boundary conditions
    #         newVoltage = self.voltageFunction(t, state, *self.voltageArgs)
    #         self.ePot_bc = {"x-": {"derivative": newVoltage}, "x+": {"value": 0}}
    #
    #     return post_step_hook, 0.0 # hook function + initial data for t

    # todo: verify this works on pickled data
    def plot_data(self, params:dict):
        """
        Plots the results of the simulation

        Args:
            params (dict): a dict of simulation parameters. The following keys are required to determine the function behavior:
            Keys:
            plotTypes (list): a list of strings for the type of plots to generate. If more than one type is listed, plots are
                generated one at a time in the order specified.
                Plot types: "kymographs", ...
            saveQ (boolean): should the plots be saved. All plots save at 300 dpi
            showPlotsQ (boolean): should the plots be displayed after they are generated
                Note: if both saveQ and showPlotsQ are both false, this function will do nothing
            saveFormat (string): format the plots are saved in. Default is .pdf
            saveName (string): name of the plots that are saved. File names are saveName_plotType.saveFormat
            saveDir (string): directory to save the plots in

        Returns:
            None (plots are shown or saved as specified)

        Note that improper inputs in the params dict will result in printed errors rather than warnings. This is to allow
        all plots that can be done to be performed without needing to restart the whole simulation.
        """

        # gather inputs in a cleaner way
        plotTypes = params["plotTypes"]
        plotArgs = params["plotArgs"]
        saveQ = params["saveQ"]
        showPlotsQ = params["showPlotsQ"]
        saveFormat = params["plotSaveFormat"]
        saveName = params["saveName"]
        saveDir = params["saveDir"]

        # generate the base filename and generate a timestamp in case there are any overlaps with existing files
        fileBaseName = os.path.join(saveDir, saveName)
        timeStamp = time.time()

        # iterate through the specified plots, displaying and saving as needed
        for plot in plotTypes:

            match plot:

                case "kymographs":
                    plot_kymographs(self.storage)

                case "concVsTime":

                    # todo: might be worth moving all of the individual plot functions to separate methods
                    # check that the proper plotArgs are specified
                    if "concVsTime" not in plotArgs.keys():
                        print("PNP.plot_data WARNING: a \"concVsTime\" plot was specified in params[\"plotTypes\"] but no corresponding" +
                                   "arguments were supplied in params[\"plotArgs\"]. Please add a key named \"concVsTime\" to params[\"plotArgs\"]." +
                                   "No concVsTime plot will be produced.")
                        continue # moves on to the next iteration of the for loop
                    plotPoints = plotArgs["concVsTime"]
                    if type(plotPoints) != list:
                        print("PNP.plot_data WARNING: A concVsTime plot requires that the value of params[\"plotArgs\"][\"concVsTime\"] be a list." +
                              "The current value is " + str(plotPoints) + ". No concVsTime plot will be produced.")
                        continue

                    # gather storage data, grid info, and generate time array
                    storageArr = np.array(self.storage.data)
                    gridArr = self.initConcFields.grid.coordinate_arrays[0]
                    numberOfStoragePoints = math.floor(params["tStop"]/params["trackerStep"]) + 1
                    tArr = np.linspace(0, params["tStop"], numberOfStoragePoints, endpoint = True)

                    # generate the plots by iterating through plotPoints and numberOfSpecies
                    for point in plotPoints:
                        for species in range(self.numberOfSpecies):
                            plotLabel = params["names"][species] + " at x = " + str(gridArr[point]) + " nm"
                            plt.plot(tArr, storageArr[:, species, point], label = plotLabel)
                    plt.legend()

            if showPlotsQ:
                plt.show()

            if saveQ:

                fileName = fileBaseName + "_" + plot + "_" + saveFormat

                # check that the file does not already exist. If it does, print a warning and save with the time appended
                if os.path.isfile(fileName):
                    fileName = os.path.join(params["saveDir"], params["saveName"]) + "_" + plot + "_" + str(
                        time.time()) + saveFormat
                    print(
                        "PNP.plot_data: WARNING - specified saveDir + saveName already exists. Saving plot as " +
                        fileName + " instead.")

                # save the file and close the plot if it wasn't displayed
                plt.savefig(fileName, dpi = 300)
                if not showPlotsQ:
                    plt.close()

    # todo: figure out if data is actually pickling. If not,the issue is probably because a method cannot pickle itself?
    #   i.e. cannot pickle(self). Consider gathering all relevant data into a dict and pickling that
    def save_data(self, params:dict, result):
        """
        Saves the results of the simulation as a pickle

        Args:
            params (dict): a dict of simulation parameters. The following keys are required to determine the function behavior:
            Keys:
            saveQ (boolean): should the simulation data be saved
            saveName (string): name of the plots that are saved. File names are saveName_data.pickle
            saveDir (string): directory to save the plots in
        """
        if "saveQ" not in params.keys() or "saveName" not in params.keys() or "saveDir" not in params.keys():
            ValueError("PNP.save_data: supplied parameter dict does not have the required keys saveQ, saveName, and saveDir.")

        if params["saveQ"]:
            # make sure the result is saved as a class variable
            self.result = result

            fileName = os.path.join(params["saveDir"], params["saveName"]) + "_data.pickle"

            # check that the file does not already exist. If it does, print a warning and save with the time appended
            if os.path.isfile(fileName):
                fileName = os.path.join(params["saveDir"], params["saveName"]) + "_data_" + str(time.time()) + ".pickle"
                print("PNP.save_data: WARNING - specified saveDir + saveName already exists. Saving simulation data as " +
                      fileName + " instead.")
            with open(fileName, 'wb') as f:
                pickle.dump(self, f)
            f.close()
        else:
            pass

    def check_index_matching(self,lists:list, keys:list):
        """
        Checks that every element of an input list of lists has the same length

        Args:
            lists (list): a list of lists taken from input parameters
            keys (list): a list of the keys associated with each list in the parameter dict

        Returns:
            boolean: True if index matching is correct, else false. Note false should never return: an exception should be raised
        """

        # check that every element of the input is a list
        for i in range(len(lists)):
            if type(lists[i]) != list:
                raise TypeError("PNP.check_index_matching: Parameter " + str(keys[i]) + " must be a list. It is currently set to "
                                + str(lists[i]) + ". Please check the value and restart the script.")

        # get the length of the first list
        len0 = len(lists[0])

        # if any other list does not match the first length, raise an error
        for i in range(len(lists)):
            if len(lists[i]) != len0:
                raise ValueError("PNP.check_index_matching: Parameters " + str(keys[0]) + " and " +
                                 str(keys[i])) + (" have different lengths and are thus not properly index matched. Please "
                                "check that all index matched keys " + str(keys) + " have the same length.")

        return True




#########################################################
############ Run Simulation ############################
#######################################################

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
    eq.save_data(params, result)
    eq.plot_data(params)
    return 0

run_simulation(params)