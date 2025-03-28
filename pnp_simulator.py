import numpy as np
from pde import PDEBase, FieldCollection, CartesianGrid, ScalarField,  MemoryStorage, movie, plot_kymographs, solve_poisson_equation
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

    ####################################################
    ####### Chemical Components ####################
    ##################################################

    # note: all lists describing the chemical species to simulate MUST have equal lengths
    "names": ["A+", "a+", "B-"],  # list of strings: names of each species in simulation
    "charges": [1, 1, -1],  # list of integers: charges of each species
    "diffusivities": [1.1 * (10 ** -5), 2.2 * (10 ** -5), 2.2 * (10 ** -5)], # list of floats: diffusion constant of each species, in cm^2/s
    "bulkConcentrations": [0.05, 0.05, 0.1],  # list of floats: concentration at far edge of simulation box, in mol/L
    "initialConcentrationProfiles": ["bulk", "bulk", "bulk"],  # list of floats or ndarrays: initial concentration profiles
                                                        # options:
                                                        # "bulk" to match the bulkConcentration of the species
                                                        # "zero" to set initial to zero
                                                        # array: set custom values at each point.
                                                        #   must be a numpy array. If the length is longer than the number of grid points,
                                                        #   extra values will be cut off. If the array is shorter than the number of grid points,
                                                        #   an IndexError will be raised
    "solventDielectric" : 1, # float: relative dielectric constant of the solvent
    "temperature": 300,  # float: temperature in K. Raising temperature increases the ratio of diffusion to drift

    #######################################################
    ############ Reactions ###############################
    ######################################################
    # Information for chemical reactions. For now these are only implemented with rate constants and orders, ignoring thermodynamic data
    #   Reactions are divided into bulkReactions (which occur between species in solution) and surfaceReactions (which only occur at the electrode surface)
    #   Bulk reactions follow Arrhenius kinetics, with an activation energy (Ea) and a reaction rate constant for T= 0 K (k0) and a Boltzmann factor
    #       dc/dt = k exp(Ea/kT) Product([conc of reactant i]**(order of reactant i))
    #   Surface reactions replace the activation energy term with equilibrium potential term (E0) in order to calculate the reaction overpotential
    #       The rest of the equation follows Arrhenius kinetics. surface reactions are only calculated for the reactants at the electrode surface. Adsorption is not currently implemented
    #       Parameters: reaction constant (k0), equil potential (E0), field at surface (E)
    #       dc/dt = k0 exp(E0-E/kT) Product([conc of reactant i]**(order of reactant i))
    # todo: implement kinetics that approximate based on gibbs energy of reaction
    # todo: add contribution from electrochemical potenial for bulk reactions
    #   todo: test if surface reaction gives mass-transfer limited Butler-Volmer Equations

    "bulkReactions" : [{}], # list of dicts: definitions for each reaction that can occur through the reaction mixture
                            #       Within each reaction, the following fields are required:
                            #           "name" (str): name to refer to the reaction
                            #           "k" (float): Rate constant of the reaction, in M**(sum(orders) -1) /s (will be converted to number/nm**3 internally)
                            #           "Ea" (float): activation energy in eV
                            #           "reactants" (list of ints): indices of each reactant
                            #           "orders" (list of floats): reaction order of each reactant, index matched to "reactants"
                            #           "stoichiometry" (list of ints): stoichiometric coefficient for each species
                            #               the calculated dc/dt at each location will be multiplied by the stoichiometry for each species to get that species' change in conc
                            #               This should include products (positive coefficients) and reactants (negative coeffs)
                            #               This must be index matched to "names"
    "surfaceReactions" : [{}], # list of dicts: definitoins for each reaction that occurs at the electrode surface
                                #       Within each reaction, the following fields are required:
                                #           "name" (str): name to refer to the reaction
                                #           "k" (float): Rate constant of the reaction, in M**(sum(orders) -1) /s (will be converted to number/nm**3 internally)
                                #           "E0" (float): equilibrium electric potential, used to calculate the overpotential
                                #           "reactants" (list of ints): indices of each reactant
                                #           "orders" (list of floats): reaction order of each reactant, index matched to "reactants"
                                #           "stoichiometry" (list of ints): stoichiometric coefficient for each species
                                #               the calculated dc/dt at each location will be multiplied by the stoichiometry for each species to get that species' change in conc
                                #               This should include products (positive coefficients) and reactants (negative coeffs)
                                #               This must be index matched to "names"

    ##################################################
    ######### Simulation Conditions ##################
    ################################################

    "voltageFunction" : vf.equilBeforeSquare, # function: must take the state and time as arguments and
                            # output a single float to be used as the applied voltage.
    "voltageFunctionArgs" : [0.05, 0.00002, 0.5, -0.5, 0.5], # list: extra arguments to pass to the voltage function, if needed
                                # if no extra arguments are needed, use an empty list
    "gridMax" : 1, # float: distance from the electrode surface to simulate, in nm
    "gridStep": 0.01,  # float: distance between grid points, in nm
    "tStep": 0.00001,  # float: time step interval for each simulation iteration, in ns
    "tStop": 0.1,  # float: time the simulation will run to. Number of steps = tStop/tStep
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
    "saveQ" : True, # boolean: should the result of the simulation and plots (if made) be pickled (saved) when the simulation finishes?
    "showPlotsQ": True,  # boolean: should the generated plots be shown after they are generated?
    "saveName" : "AaB_mixedcation_equil_squarewave_05equil_05duty_05amp_-05offset_200fsperiod_phaseCorrection", # string: the base name of the simulation to save associated date. File names are saveName_dataType.saveFormat
    "saveDir" : "C://Users//shams//Drexel University//Chang Lab - General//Individual//Sam Amsterdam//transient electrochemistry//pnp-simulator//", # string: the directory to save data in
    "plotTypes" : ["kymographs", "concVsTime"], # list of strings: what types of plots should be generated.
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
        evolution_rate(state, t) -- the function that is run every iteration of the simulation in order to advance. The
            voltage boundary conditions are also updated in this function. This is not directly called by the user but is used by PDEBase.solve()
        update_rxns(concs, ePot) -- calculates the dc/dt due to user-specified reactions. This is called during evolution_rate
            but is moved to a separate function for readability
        solve(fields, t_range, dt, tracker) -- PDEBase method used to run the simulation. It is not modified here but is
            called by the run_simulation function elsewhere in the script
        plot_data(plotTypes, saveQ, saveName, saveDir) -- plots the results of the simulation
        check_index_matching(lists, keys) -- checks that a list of lists all have the same length, raises an exception otherwise

        Init does initialize the saveVars class variable, which allows the clas to track which variables are saved and loaded
        between uses
        """
        self.saveVars = ["inputParams", "result", "storage", "ePot_bc"]

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
        self.inputParams = params
        self.dt = params["tStep"]
        self.numberOfPoints = math.floor(params["gridLen"]/params["gridStep"]) # math.floor used to ensure numberOfPoints is an integer

        # calculate values that will be used in every simulation iteration and save them as class variables
        # lists are converted to arrays in order to multiply by floats
        self.dielectric = params["solventDielectric"] * params["vacuumPermitivitty"] # absolute solvent dielectric constant in e^2/eVnm
        self.diffusivities = (10**5) * np.array(params["diffusivities"]) # convert diffusivities from cm^2/s to nm^2/ns
        self.kt = params["kb"] * params["temperature"] # boltzmann constant * temperature in eV
        self.bulk_concs = (params["Av"] / (10**24)) * np.array(params["bulkConcentrations"]) # bulk concentrations converted from mol/L to number / nm^3
        self.zs = np.array(params["charges"])
        self.numberOfSpecies = len(self.zs) # number of species comes up a lot later

        # prep reaction parameters: pre-calculate Arrhenius factors, check for index matching
        self.bulkRxns = self.params["bulkReactions"]
        self.surfRxns = self.params["surfaceReactions"]
        for rxn in self.bulkRxns:
            self.check_index_matching([rxn["reactants"], rxn["orders"]], ["reactants", "orders"])
            self.check_index_matching([params["names"], rxn["stoichiometry"]], ["names", "stoichiometry"])
            rxn["prefactor"] = rxn["k"] * math.exp(rxn["Ea"] / self.kt)
        for rxn in self.surfRxns:
            self.check_index_matching([rxn["reactants"], rxn["orders"]], ["reactants", "orders"])
            self.check_index_matching([params["names"], rxn["stoichiometry"]], ["names", "stoichiometry"])

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
        self.grid = CartesianGrid([(0, params["gridMax"])], [self.numberOfPoints], False)

        # for each species, create a field using the corresponding init profile
        concArrays = []
        for i in range(self.numberOfSpecies):

            initProfile = params["initialConcentrationProfiles"][i]

            # if "bulk" match bulk conc
            if initProfile == "bulk":
                profileArray = np.ones(self.numberOfPoints) * self.bulk_concs[i]

            # if "zero" set to 0
            elif initProfile == "zero":
                profileArray = np.zeros(self.numberOfPoints)

            # if a custom array, first check that it is the correct length (raise error if not), than create
            elif type(initProfile) == np.ndarray:

                # handle malformed inputs
                if len(initProfile) < self.numberOfPoints:
                    IndexError("PNP.initialize_fields: a supplied initial concentration profile for species " +
                               str(params["names"][i])) + (" is shorter than the grid length specified by gridMax and "
                                                           "gridStep. Simulation cannot run until this is addressed.")
                elif len(initProfile) > self.numberOfPoints:
                    print("PNP.initialize_fields: a supplied initial concentration profile for species " +
                               str(params["names"][i])) + (" is longer than the grid length specified by gridMax and "
                                                           "gridStep. Excess values beyond the grid length will be ignored.")

                profileArray = initProfile[:self.numberOfPoints] # slice cuts off excess values. Warning about this situation is raised above

            concArrays.append(profileArray)

        # gather all concArrays into a FieldCollection and return
        return FieldCollection([ScalarField(self.grid, concArray) for concArray in concArrays], labels = params["names"])

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

        # calculate reaction changes
        rxn = self.update_rxns(concs, ePotential)

        dcdt = [drift[i] + diff[i] + rxn[i] for i in range(self.numberOfSpecies)]

        # calculate the voltage for the next step
        newVoltage = self.voltageFunction(t + self.dt, state, *self.voltageArgs)
        self.ePot_bc = {"x-": {"derivative": newVoltage}, "x+": {"value": 0}}

        return FieldCollection(dcdt)

    # todo: add documentation to __init__
    def update_rxns(self, concs, ePot):
        """
        Calculates the dc/dt for each species based on the reactions specified in the input parameters

        Args:
            concs (FieldCollection): the current state of the simulation
            ePot (ScalarField): the electric potential of the simulation (from solving Poisson's Equation)
        Returns:
            list of ScalarFields: the dc/dt of each species due to the sum of all bulk reactions
        """
        # gather the data arrays from the conc fields as a vertical stack of arrays
        # dimensions are grid len x number of species
        concArr = np.zeros((self.numberOfSpecies, self.numberOfPoints))
        for i in range(self.numberOfSpecies):
            concArr[i,:] = concs[i].data

        # construct an array to hold the dc/dt for each species
        totalRxndcdt = np.zeros((self.numberOfSpecies, self.numberOfPoints))

        # first handle the bulk reactions
        # todo: there are a lot of duplicated variable names, makes sure there are no aliasing issues
        for rxn in self.bulkRxns:

            # gather rxn info
            reactants = rxn["reactants"]
            orders = rxn["orders"]
            prefactor = rxn["prefactor"]

            # construct an array out of the order-weighted concentrations of each reactant
            orderConcs = np.zeros((len(reactants), self.numberOfPoints))
            for i in range(len(reactants)):
                # note: reactants[i] is the row of the species in concArr
                orderConcs[i,:] = concArr[reactants[i],:] ** orders[i]

            # calculate the overall non-stoichiometric dc/dt by multiplying every element in the stack along with the prefactor
            nonStoichdcdt = prefactor * np.prod(orderConcs, axis = 0)

            # iterate through the stoichiometry, add the stoichiometric dc/dt to the total dc/dt array at the correct row
            for i in range(self.numberOfSpecies):
                factor = rxn["stoichiometry"][i]
                if factor == 0: # the compiler would probably do this optimization on its own, but whatever
                    pass
                else:
                    totalRxndcdt[i,:] += factor * nonStoichdcdt

        # next handle the surface reactions in a similar way but only working with the first elements of each conc field
        for rxn in self.surfRxns:

            # gather rxn info
            reactants = rxn["reactants"]
            orders = rxn["orders"]
            # todo: check signs on overpotential in anodic vs cathodic rxn
            overpotential = ePot.data[0] - rxn["E0"]
            prefactor = overpotential / self.kt

            # construct an array of the order-weighted concentrations of each reactant at the surface
            # since this data is much smaller than the full array, we can work directly with lists instead
            orderConcs = [concArr[reactants[i],0] ** orders[i] for i in range(len(reactants))]

            # calculate the non-stoichiometric dc/dt by multiplying every element in the list with the prefactor
            nonStoichdcdt = math.prod(orderConcs) * prefactor

            # iterate through stoichiometry, add the stoichiometric dc/dt to the first element in the corresponding row of the total dc/dt
            for i in range(self.numberOfSpecies):
                factor = rxn["stoichiometry"][i]
                totalRxndcdt[i,0] += factor * nonStoichdcdt

        # return a list of the dc/dt as scalar fields
        return [ScalarField(self.grid, totalRxndcdt[i,:]) for i in range(self.numberOfSpecies)]

    # todo: implement movies (need to make sure ffmpeg is in sys.path)
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
        plotTypes = self.inputParams["plotTypes"]
        plotArgs = self.inputParams["plotArgs"]
        saveQ = self.inputParams["saveQ"]
        showPlotsQ = self.inputParams["showPlotsQ"]
        saveFormat = self.inputParams["plotSaveFormat"]
        saveName = self.inputParams["saveName"]
        saveDir = self.inputParams["saveDir"]

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
                    numberOfStoragePoints = math.floor(self.inputParams["tStop"]/self.inputParams["trackerStep"]) + 1
                    tArr = np.linspace(0, self.inputParams["tStop"]/self.inputParams["tStep"], numberOfStoragePoints, endpoint = True)

                    # generate the plots by iterating through plotPoints and numberOfSpecies
                    for point in plotPoints:
                        for species in range(self.numberOfSpecies):
                            plotLabel = self.inputParams["names"][species] + " at x = " + str(gridArr[point]) + " nm"
                            plt.plot(tArr, storageArr[:, species, point], label = plotLabel)
                    plt.legend()

            if saveQ:

                fileName = fileBaseName + "_" + plot + "_" + saveFormat

                # check that the file does not already exist. If it does, print a warning and save with the time appended
                if os.path.isfile(fileName):
                    fileName = os.path.join(self.inputParams["saveDir"], self.inputParams["saveName"]) + "_" + plot + "_" + str(
                        time.time()) + saveFormat
                    print(
                        "PNP.plot_data: WARNING - specified saveDir + saveName already exists. Saving plot as " +
                        fileName + " instead.")

                # save the file and close the plot if it wasn't displayed
                if plot == "kymographs":
                    # kymographs option action = "close" saves the plot
                    plot_kymographs(self.storage, filename = fileName, action = "close", fig_style = {'dpi':300})
                else:
                    plt.savefig(fileName, dpi = 300)
                    if not showPlotsQ:
                        plt.close()

            if showPlotsQ:
                plt.show()

    def save_data(self):
        """
        Saves the results of the simulation as a pickle, which can then be loaded and analyzed or rerun later

        Args:
            None

        The following information specified by self.saveVars is gathered in a dict and pickled:
            self.inputParams
            self.result
            self.storage
            self.ePot_bc
        """
        if "saveName" not in params.keys() or "saveDir" not in params.keys():
            ValueError(
                "PNP.save_data: supplied parameter dict does not have the required keys saveQ, saveName, and saveDir.")

        # gather data to pickle
        # note that ePot_bc is saved because it changes during the simulation. This allows the final form to be recovered
        #   without re-running the experiment. I can't think of a good reason to do this, but for the sake of robustness why not?
        saveDat = {'inputParams':self.inputParams, 'result':self.result, 'storage':self.storage, 'ePot_bc': self.ePot_bc}

        fileName = os.path.join(self.inputParams["saveDir"], self.inputParams["saveName"]) + "_data.pickle"

        # check that the file does not already exist. If it does, print a warning and save with the time appended
        if os.path.isfile(fileName):
            fileName = os.path.join(self.inputParams["saveDir"], self.inputParams["saveName"]) + "_data_" + str(time.time()) + ".pickle"
            print("PNP.save_data: WARNING - specified saveDir + saveName already exists. Saving simulation data as " +
                  fileName + " instead.")
        with open(fileName, 'wb') as f:
            pickle.dump(saveDat, f)
        f.close()

    def load_data(self, pickleFile):
        """
        Loads pickled data into an empty PNP class instance.

        Args:
            pickleFile (str): the file path and name of the pickle to load

        Returns:
            None

        If the PNP class instance was previously empty, all of its class variables will be populated. Normal plotting and
        analysis methods can be used on the loaded class instance
        """
        # check if the PNP instance was previously initialized by checking if any of the save variables already exist in the PNP class
        for saveVar in self.saveVars:
            if hasattr(self, saveVar):
                ValueError("PNP.load_data: attempting to load a pickle into a PNP instance that has already been populated.\n"
                           "To avoid overwriting data, this action is blocked. Instead create a new instance of the PNP class and load into that:\n"
                           "newInstance = PNP()\n"
                           "newInstance.load_data(file)")

        # open the pickle
        with open(pickleFile, 'rb') as f:
            pnpDict = pickle.load(f)
        f.close()

        # check that the loaded file is a dict with the correct keys
        if type(pnpDict) != dict:
            TypeError("PNP.load_data: the loaded pickle was not a dict. Loading aborted, check that the file name is correct.")
        for saveVar in self.saveVars:
            if saveVar not in pnpDict.keys():
                KeyError("PNP.load_data: the loaded pickle is missing data key " + saveVar + ". Loading is not possible."
                         "Check that the file name is correct and that the data was created by PNP.save_data().")

        # use unpack_parameters to set up the initial data. this must be done before re-saving self.storage and self.ePot_bc to avoid overwriting the saved versions
        self.unpack_parameters(pnpDict["inputParams"])

        # save/overwrite result, storage, and ePot_bc
        self.result = pnpDict["result"]
        self.storage = pnpDict["storage"]
        self.ePot_bc = pnpDict["ePot_bc"]

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
    eq.result = eq.solve(*solveParams)
    if params["saveQ"]:
        eq.save_data()
    eq.plot_data(params)
    return 0

run_simulation(params)

# path = "C://Users//shams//Drexel University//Chang Lab - General//Individual//Sam Amsterdam//transient electrochemistry//pnp-simulator//"
# file = path + "nacl_testing__data.pickle"
# eq = PNP()
# eq.load_data(file)
# eq.plot_data(eq.params)
#
# print('done')