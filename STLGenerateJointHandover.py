#!/usr/bin/env python
from STL import STLFormula
import operator as operatorclass
import pulp as plp
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
import random
import time
from sklearn.metrics import mean_squared_error as mse
from tfmatrix import points2tfmatrix, points2invtfmatrix


# Author: Jonathan Fredberg
# Edited: Parag Khanna

# Based on implementation by Alexis Linard



class generate_signal_problem:

    def __init__(self, phi, start, domain, dimensions, U, dt, OPTIMIZE_ROBUSTNESS = True, QUANTITATIVE_OPTIMIZATION = True, HARD_STL_CONSTRAINT = False, KSHIRSAGAR_APPROACH = True, KSHIRSAGAR_DELTA = [0.3, 0.1, 0.0], MINIMIZE_MAX_VELOCITY = False, MINIMIZE_AVG_VELOCITY = False , Kshirsagar_phi = None, HARD_KSHIRSAGAR_CONSTRAINT = False):
        """
            Provides methods for generating a signal satisfying an STL Formula.
            Takes as input:
                * phi: an STL Formula
                * start: a vector of the form [x0,y0,...] for the starting point coordinates
                * domain: the domain on which signals are generated. self.domain = [lb,ub] where lb is the lower bound and ub the upper bound of the domain.
                * dimensions: the dimensions on which the STLFormula is defined,
                    must be: ['giver_x','giver_y','giver_z','giver_dx','giver_dy','giver_dz', 'taker_x', 'taker_y', 'taker_z', 'taker_dx', 'taker_dy', 'taker_dz', 'relative_x','relative_y','relative_z','K_p']
                * U: a basic control policy standing for maximum speed and acceleration of giver and taker, i.e. \forall t \in [0,T], |s[t]-s[t+1]| < U \pm \epsilon 
                    defined as:     max_change = [giver_max_speed_x * dt, giver_max_speed_y * dt, giver_max_speed_z * dt, giver_max_acceleration_x * dt, giver_max_acceleration_y * dt, giver_max_acceleration_z * dt, 
                                                  taker_max_speed_x * dt, giver_max_speed_y * dt, giver_max_speed_z * dt, giver_max_acceleration_x * dt, giver_max_acceleration_y * dt, giver_max_acceleration_z * dt]
                * dt: sampling time of system
                * OPTIMIZE_ROBUSTNESS: a flag whether the robustness of the generated signal w.r.t. phi has to be maximized or not. i.e. negative robustness is minimized as part of objective function
                * HARD_STL_CONSTRAINT: a flag whether STL constraints MUST be fullfilled, or if negative robustness is penalized. Can allow for breaking constraints if fullfilling is impossible.
                    i.e. True -> robustness >= 0 is constraint;     False -> max(0, -robustness) * very_large_number is part of objective function
                * KSHIRSAGAR_APPROACH: a flag whether phi contains the l1 distance norm, 'K_p' between giver and taker based on the "towards human hand" approach method in a paper by Kshirsagar et. al [FLAGFLAGFLAG I should add citation to that]
                * MINIMIZE_AVG_VELOCITY: a flag whether average velocity should be part of the objective function, computed as sum of l1 norm of all timesteps
                * MINIMIZE_MAX_VELOCITY: a flag whether maximum velocity should be part of the objective function, computed as maximum l1 norm

            The encoding details of the MILP optimization problem follows the quantitative enconding of Raman et al., "Model  predictive  control  with  signaltemporal logic specifications" in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.
            FLAGFLAGFLAGFLAGFLAG!!!!!!!!!!! Write credit to Kshirsagar for aproach STL
        """ 
        #CONSTANTS
        self.M = 100000
        self.M_up = 100000
        self.M_low = 0.000001
        
        # Values
        self.phi = phi
        self.start = start
        self.domain = domain 
        self.dimensions = dimensions
        self.U = U
        self.dt = dt
        self.Kshirsagar_phi = Kshirsagar_phi

        self.KSHIRSAGAR_APPROACH = KSHIRSAGAR_APPROACH
        self.HARD_KSHIRSAGAR_CONSTRAINT = HARD_KSHIRSAGAR_CONSTRAINT
        self.KSHIRSAGAR_DELTA = KSHIRSAGAR_DELTA
        self.OPTIMIZE_ROBUSTNESS = OPTIMIZE_ROBUSTNESS
        self.QUANTITATIVE_OPTIMIZATION = QUANTITATIVE_OPTIMIZATION
        self.HARD_STL_CONSTRAINT = HARD_STL_CONSTRAINT
        self.MINIMIZE_MAX_VELOCITY = MINIMIZE_MAX_VELOCITY
        self.MINIMIZE_AVG_VELOCITY = MINIMIZE_AVG_VELOCITY

        if QUANTITATIVE_OPTIMIZATION:
            self.init_quantitave_problem()
        else:
            print("Boolean satisfaction solver not implemented yet")
            raise

        self.mse_giver = None
        self.mse_taker = None


    def fix_variables(self, t, variable_idx, value=False):
        """
            Sets variables in self.s to simplify solving the problem.
            Fixing a variable sets upper and lower bounds to the value of the variable.
            Inputs:
                t:              List of int Time index for all variables to set
                variable_idx:   List of int 2nd index of variables to set
                value:          List of lists containing Values to fix, value[time][variable]
        """
        
        # If value is set 
        if not(value == False):
            try:    # Try to set init value for vatiable
                [[self.s[t[j]][variable_idx[i]].setInitialValue( value[j][i] ) for i in range(len(variable_idx))] for j in range(len(t))]
            except ValueError:
                print("WARNING: Trying to fix allready fixed variables t=" + str(t) + ", i=" + str(variable_idx) + "\n " + str(ValueError) + "\n call self.unfix_variables before rewriting allready fixed variables!\n Unfixing and retrying")
                self.unfix_variables(t, variable_idx)
                [[self.s[t[j]][variable_idx[i]].setInitialValue( value[j][i] ) for i in range(len(variable_idx))] for j in range(len(t))]

        # Fix the variable to current value
        [[self.s[j][i].fixValue() for i in variable_idx] for j in t]

    def unfix_variables(self, t, variable_idx):
        """
            Unfixes variables in self.s, so value can be changed
            Inputs:
                t:              List of int Time index for all variables to set
                variable_idx:   List of int 2nd index of variables to set
        """
        [[self.s[j][i].unfixValue() for i in variable_idx] for j in t]

    def get_max_acceleration(self, times, idxs):
        """
            Calculates the maximum acceleration from values in s, for trubleshooting
            Output:
                
        """
        max_accelerations = []
        max_speed = []
        try:
            for i in idxs:
                x = [self.s[t][i].varValue for t in times]
                dx = [(x[t+1] - x[t])/self.dt for t in range(len(times)-1)]
                ddx = [(dx[t+1] - dx[t])/self.dt for t in range(len(times)-2)]
                max_speed.append(max([abs(a) for a in dx]))
                max_accelerations.append(max([abs(a) for a in ddx]))
                      #(self.s[t+1][i] - self.s[t][i])/ self.dt
        except:
            pass
        return max_accelerations

    def set_leading_states(self, current_giver_shared, current_taker_shared):
        """
            Updates values for states for times t <= 0
            used in state 1, 4 of the state machine
            
            # update past states s[-2]=s[-1]
            # update past states s[-1]=s[0]
            # set current state  s[0]=current positions

            Input:
                current_giver_shared    list, [x,y,z] of current position of giver in shared frame
                current_taker_shared    list, [x,y,z] of current position of taker in shared frame
        """
        # Get idx
        giver_pos_idx = [self.dimensions.index('giver_x'),self.dimensions.index('giver_y'),self.dimensions.index('giver_z')]
        taker_pos_idx = [self.dimensions.index('taker_x'),self.dimensions.index('taker_y'),self.dimensions.index('taker_z')]
        
        # Get previous values
        previous_giver_0  = [self.s[0][i].varValue  for i in giver_pos_idx]
        previous_giver_n1 = [self.s[-1][i].varValue for i in giver_pos_idx]
        previous_taker_0  = [self.s[0][i].varValue  for i in taker_pos_idx]
        previous_taker_n1 = [self.s[-1][i].varValue for i in taker_pos_idx]

        # Update -2 (if it is defined, and -1 as previously been fixed)
        if (not(previous_giver_n1[0] is None)) and self.s[-1][giver_pos_idx[0]].isFixed():
            self.unfix_variables([-2], 
                               giver_pos_idx)
            self.unfix_variables([-2], 
                               taker_pos_idx)

            self.fix_variables([-2], 
                               giver_pos_idx,
                               [previous_giver_n1])
            self.fix_variables([-2], 
                               taker_pos_idx,
                               [previous_taker_n1])

        # Update -1 (if it is defined, and 0 as previously been fixed)
        if (not(previous_giver_0[0] is None)) and self.s[0][giver_pos_idx[0]].isFixed():
            self.unfix_variables([-1], 
                               giver_pos_idx)
            self.unfix_variables([-1], 
                               taker_pos_idx)

            self.fix_variables([-1], 
                               giver_pos_idx,
                               [previous_giver_0])
            self.fix_variables([-1], 
                               taker_pos_idx,
                               [previous_taker_0])

        # Update 0
        self.unfix_variables([0], 
                               giver_pos_idx)
        self.unfix_variables([0], 
                               taker_pos_idx)

        self.fix_variables([0], 
                           giver_pos_idx,
                           [current_giver_shared])
        self.fix_variables([0], 
                           taker_pos_idx,
                           [current_taker_shared])

        
        return True

    def generate_path(self, current_step):
        """
            Generates a path following self.opt_model
            Returns self.s: The optimized signal 
        """
        warm_ex_start = time.monotonic()
        success = self.model_dict[current_step].solve(plp.GUROBI_CMD(msg=False,  warmStart=False)) # FLAGFLAGFLAG Here be terminal messages
        #success = self.opt_model.solve() # FLAGFLAGFLAG Here be terminal messages
        warm_ex_time = time.monotonic() - warm_ex_start
        
        #self.model_dict[current_step].writeLP("model_t"+ str( current_step ) +".lp")

        # Output robustness to text file
        #txtout = open("robustness.txt", "a")
        #obj = str(self.model_dict[current_step].objective).replace('-','')
        #txtout.write("t" + str( current_step ) + ";    "+ obj + " = " + str( -plp.value(self.model_dict[current_step].objective) ) + "\n")
        #txtout.close()

        print("Generate path Gurobi time    = " + str(round(warm_ex_time, 5)) + "\n")

        if success:
            print("Plan found")
            return [[self.s[j][i].varValue for i in range(len(self.dimensions))] for j in range(self.phi.horizon+1)]
        else:
            print("Planner failed")
            return success

    def get_path(self):
        """
            Returns the path
        """
        return [[self.s[j][i].varValue for i in range(len(self.dimensions))] for j in range(self.phi.horizon+1)]

    def get_K(self, n):
        """
            Returns the value of K at time step n
        """
        return self.s[n][self.dimensions.index('K_p')].varValue

    def get_predicted_handover_step(self, K_epsilon, current_step):
        """
            Returns the index of first time step from current_step to end of handover where the handover distance K < 0.1
            If no satisfactory handover is found, the time for lowest K is found
        """
        K_list = [self.s[n][self.dimensions.index('K_p')].varValue for n in range(current_step, self.phi.horizon+1)]

        # Directly calculating K as the l1 norm of giver - taker + delta to compare to the encoding. Confirms
        expected_K_list = [ abs( self.s[n][self.dimensions.index('giver_x')].varValue - self.s[n][self.dimensions.index('taker_x')].varValue + self.KSHIRSAGAR_DELTA[0]) + abs( self.s[n][self.dimensions.index('giver_y')].varValue - self.s[n][self.dimensions.index('taker_y')].varValue + self.KSHIRSAGAR_DELTA[1]) + abs( self.s[n][self.dimensions.index('giver_z')].varValue - self.s[n][self.dimensions.index('taker_z')].varValue + self.KSHIRSAGAR_DELTA[2] ) for n in range(current_step, self.phi.horizon+1) ]
        K_error = [K_list[n] - expected_K_list[n] for n in range(self.phi.horizon+1 - current_step)]
        if max(K_error) > 1.0e-7:
            print("Encoding error in K!")

        try:
            handover_time = next(x[0] for x in enumerate(K_list) if x[1] < K_epsilon) + current_step
        except StopIteration:
            handover_time = K_list.index(min(K_list)) + current_step
            print("No handover location detected, closest: " + str(handover_time) )
        #handover_distance = self.get_K(handover_time)
        
        return handover_time

    def get_minimum_K(self):
        """
            Returns the minimum K distance between giver and taker, and the index when it ocurs
            Output:
                [minK, idx]
        """
        K_list = [self.s[n][self.dimensions.index('K_p')].varValue for n in range(0, self.phi.horizon+1)]

        expected_K_list = [ abs( self.s[n][self.dimensions.index('giver_x')].varValue - self.s[n][self.dimensions.index('taker_x')].varValue + self.KSHIRSAGAR_DELTA[0]) + abs( self.s[n][self.dimensions.index('giver_y')].varValue - self.s[n][self.dimensions.index('taker_y')].varValue + self.KSHIRSAGAR_DELTA[1]) + abs( self.s[n][self.dimensions.index('giver_z')].varValue - self.s[n][self.dimensions.index('taker_z')].varValue + self.KSHIRSAGAR_DELTA[2] ) for n in range(0, self.phi.horizon+1) ]
        K_error = [K_list[n] - expected_K_list[n] for n in range(0,self.phi.horizon+1)]
        if max(K_error) > 1.0e-7:
            print("Encoding error in K!")

        minK = min(K_list)

        idx = K_list.index(minK) + 2

        return [minK, idx]


    def init_quantitave_problem(self):
        """
            Function initiating a MILP optimization problem, generating a signal satisfying an STL Formula.
            
            The encoding details of the MILP optimization problem follows the quantitative enconding of Raman et al., "Model  predictive  control  with  signaltemporal logic specifications" in 53rd IEEE Conference on Decision and Control. IEEE, 2014, pp. 81–87.
        """    

        # Start of timer
        # build_model_start_time = time.monotonic()
        
        giver_x_idx = self.dimensions.index('giver_x')
        giver_dx_idx = self.dimensions.index('giver_dx')
        giver_ddx_idx = self.dimensions.index('giver_ddx')
        taker_x_idx = self.dimensions.index('taker_x')
        taker_dx_idx = self.dimensions.index('taker_dx')
        taker_ddx_idx = self.dimensions.index('taker_ddx')
        relative_x_idx = self.dimensions.index('relative_x')
        relative_dx_idx = self.dimensions.index('relative_dx')
        relative_ddx_idx = self.dimensions.index('relative_ddx')
        #if self.KSHIRSAGAR_APPROACH:
        K_p_idx = self.dimensions.index('K_p')

        self.dict_vars = {}

        #objective, maximize robustness
        for eval_t in range(0,self.phi.horizon):
            rvar = plp.LpVariable('R_'+str(id(self.phi))+'_t0_'+str(0)+"_t_"+str(eval_t),cat='Continuous')
            self.dict_vars['R_'+str(id(self.phi))+'_t0_'+str(0)+"_t_"+str(eval_t)] = rvar
            
        #Initialize model
        #self.opt_model = plp.LpProblem("MIP_Model")
        # Initialize one model for each time step
        self.model_dict = dict( [[ eval_t , plp.LpProblem("MIP_Model_t"+str(eval_t))] for eval_t in range(self.phi.horizon)] )

    
        #The signal we want to optimize. The lower and upperbounds are specified by self.domain
        # s starts at s[-2], 2 timesteps before t_0
        self.s = plp.LpVariable.dicts("s",(range(-2, self.phi.horizon+1),range(len(self.dimensions))),self.domain[0],self.domain[1],plp.LpContinuous)
        #self.s = plp.LpVariable.dicts("s",(range(self.phi.horizon+1),range(len(self.dimensions))),cat=plp.LpContinuous)

        # Set domain of signals, max and min from learning data
        #          Name             Low          Up    
        #   _________________    ________    _________

        #   "x_taker_RHand"      0.084439      0.50181
        #   "y_taker_RHand"      -0.11043      0.35285
        #   "z_taker_RHand"      -0.39868      0.17914
        #   "dx_taker_RHand"      -1.2856      0.43672
        #   "dy_taker_RHand"      -1.0515      0.57052
        #   "dz_taker_RHand"       -1.845       1.4483
        #   "ddx_taker_RHand"     -3.6179         3.43
        #   "ddy_taker_RHand"     -3.1071       2.4208
        #   "ddz_taker_RHand"     -4.4979       5.8674
        #   "x_giver_RHand"      -0.40939    -0.055837
        #   "y_giver_RHand"      -0.37347     0.081834
        #   "z_giver_RHand"      -0.41972      0.17645
        #   "dx_giver_RHand"     -0.43335      0.65061
        #   "dy_giver_RHand"     -0.18467      0.71537
        #   "dz_giver_RHand"     -0.43489       1.1723
        #   "ddx_giver_RHand"     -1.9746       1.9046
        #   "ddy_giver_RHand"     -1.2433        1.701
        #   "ddz_giver_RHand"     -2.3685       2.5063
        #for t in range(-2,self.phi.horizon+1):
        #    self.s[t][taker_x_idx].lowBound   =   0.081001
        #    self.s[t][taker_x_idx].upBound    =    0.54383
        #    self.s[t][taker_x_idx+1].lowBound   = -0.11043
        #    self.s[t][taker_x_idx+1].upBound    =  0.35285
        #    self.s[t][taker_x_idx+2].lowBound   = -0.41089
        #    self.s[t][taker_x_idx+2].upBound    =  0.15823
        #    #self.s[t][taker_dx_idx].lowBound  =    -1.2856
        #    #self.s[t][taker_dx_idx].upBound   =    0.43672
        #    #self.s[t][taker_dx_idx+1].lowBound  =  -1.0515
        #    #self.s[t][taker_dx_idx+1].upBound   =  0.57052
        #    #self.s[t][taker_dx_idx+2].lowBound  =  -1.0664
        #    #self.s[t][taker_dx_idx+2].upBound   =   1.5533
        #    #self.s[t][taker_ddx_idx].lowBound =    -3.6179
        #    #self.s[t][taker_ddx_idx].upBound  =       3.43
        #    #self.s[t][taker_ddx_idx+1].lowBound =  -3.1071
        #    #self.s[t][taker_ddx_idx+1].upBound  =   2.4208
        #    #self.s[t][taker_ddx_idx+2].lowBound =   -3.9964
        #    #self.s[t][taker_ddx_idx+2].upBound  =   3.8358
        #    self.s[t][giver_x_idx].lowBound   =     -0.41439
        #    self.s[t][giver_x_idx].upBound    =     -0.038686
        #    self.s[t][giver_x_idx+1].lowBound   = -0.37347
        #    self.s[t][giver_x_idx+1].upBound    = 0.080492
        #    self.s[t][giver_x_idx+2].lowBound   = -0.34843
        #    self.s[t][giver_x_idx+2].upBound    =  0.17645
        #    #self.s[t][giver_dx_idx].lowBound  =   -0.59365
        #    #self.s[t][giver_dx_idx].upBound   =    0.65061
        #    #self.s[t][giver_dx_idx+1].lowBound  = -0.34532
        #    #self.s[t][giver_dx_idx+1].upBound   =  0.63908
        #    #self.s[t][giver_dx_idx+2].lowBound  = -0.97647
        #    #self.s[t][giver_dx_idx+2].upBound   =   1.1723
        #    #self.s[t][giver_ddx_idx].lowBound =    -1.9746
        #    #self.s[t][giver_ddx_idx].upBound  =     1.9046
        #    #self.s[t][giver_ddx_idx+1].lowBound =  -1.7796
        #    #self.s[t][giver_ddx_idx+1].upBound  =       1.5185
        #    #self.s[t][giver_ddx_idx+2].lowBound =  -2.3832
        #    #self.s[t][giver_ddx_idx+2].upBound  =   2.5063

        # PROBABLY not necessary when initiating the problem
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #the start position is specified
        #self.fix_variables([0], [i + giver_x_idx for i in [0,1,2]], [self.start[0:3]]) # Giver position
        #self.fix_variables([0], [i + taker_x_idx for i in [0,1,2]], [self.start[3:6]]) # Taker position
        # Start velocity can't be specified, as they will conflict with motion model constraints
        
        # Set model constraint
        # These constrain the evolution of the states
        # giver position x:     s[t][0]
        # giver velocity x:     s[t][3]
        # giver accel    x:     s[t][6]
        # taker position x:     s[t][9]
        # taker velocity x:     s[t][12]
        # taker accel    x:     s[t][15]
        for eval_t in range(0,self.phi.horizon): 
            for t in range(0,self.phi.horizon+1):
                for i in [0,1,2]:
                    # Velocity[t] == (Position[t+1] - Position[t]) / dt
                    #self.model_dict[eval_t] += self.s[t][i+giver_dx_idx] == (self.s[t+1][i+giver_x_idx] - self.s[t][i+giver_x_idx])/ self.dt
                    #self.model_dict[eval_t] += self.s[t][i+taker_dx_idx] == (self.s[t+1][i+taker_x_idx] - self.s[t][i+taker_x_idx])/ self.dt
                    self.model_dict[eval_t] += self.s[t][i+giver_dx_idx] == (self.s[t][i+giver_x_idx] - self.s[t-1][i+giver_x_idx])/ self.dt
                    self.model_dict[eval_t] += self.s[t][i+taker_dx_idx] == (self.s[t][i+taker_x_idx] - self.s[t-1][i+taker_x_idx])/ self.dt

                    # Acceleration[t] == (Velocity[t+1] - Velocity[t]) / dt
                    #self.model_dict[eval_t] += self.s[t][i+giver_ddx_idx] == (self.s[t+1][i+giver_dx_idx] - self.s[t][i+giver_dx_idx])/ self.dt
                    #self.model_dict[eval_t] += self.s[t][i+taker_ddx_idx] == (self.s[t+1][i+taker_dx_idx] - self.s[t][i+taker_dx_idx])/ self.dt
                    self.model_dict[eval_t] += self.s[t][i+giver_ddx_idx] == (self.s[t][i+giver_x_idx] - 2*self.s[t-1][i+giver_x_idx] + self.s[t-2][i+giver_x_idx]) / (self.dt**2)
                    self.model_dict[eval_t] += self.s[t][i+taker_ddx_idx] == (self.s[t][i+taker_x_idx] - 2*self.s[t-1][i+taker_x_idx] + self.s[t-2][i+taker_x_idx]) / (self.dt**2)
                    
            #t = self.phi.horizon
            #for i in [0,1,2]:
            #    # Velocity[end] == Velocity[end-1]
            #    self.model_dict[eval_t] += self.s[t][i+giver_dx_idx] == self.s[t-1][i+giver_dx_idx]
            #    self.model_dict[eval_t] += self.s[t][i+taker_dx_idx] == self.s[t-1][i+taker_dx_idx]
            #    # Acceleration[end] == Acceleration[end-1]
            #    self.model_dict[eval_t] += self.s[t][i+giver_ddx_idx] == self.s[t-1][i+giver_ddx_idx]
            #    self.model_dict[eval_t] += self.s[t][i+taker_ddx_idx] == self.s[t-1][i+taker_ddx_idx]

    
        #basic control policy, i.e. max change of speed in 1 step. (U = max abs acceleration)
        # The control constraint is placed only on future states, s^(t'), for t' > t (current time)
        # Therefore for problems evaluated at time eval_t, the constraints are placed of s[t']
        #   from next step, t' = eval_t+1, to end of signal, t' = phi.horizon+1
        for eval_t in range(0,self.phi.horizon): 
            for t in range(eval_t+1,self.phi.horizon+1):    
                for i in [0,1,2]:
                    #self.model_dict[eval_t] +=   self.s[t+1][i+giver_dx_idx]-self.s[t][i+giver_dx_idx]  <= self.U[i+giver_dx_idx] # Giver positive acceleration limit
                    #self.model_dict[eval_t] += -(self.s[t+1][i+giver_dx_idx]-self.s[t][i+giver_dx_idx]) <= self.U[i+giver_dx_idx] # Giver negative acceleration limit
                    #self.model_dict[eval_t] +=   self.s[t+1][i+taker_dx_idx]-self.s[t][i+taker_dx_idx]  <= self.U[i+taker_dx_idx] # Taker positive acceleration limit
                    #self.model_dict[eval_t] += -(self.s[t+1][i+taker_dx_idx]-self.s[t][i+taker_dx_idx]) <= self.U[i+taker_dx_idx] # Taker negative acceleration limit
                    
                    self.model_dict[eval_t] +=   self.s[t][i+giver_ddx_idx]  <= self.U[i]
                    self.model_dict[eval_t] += -(self.s[t][i+giver_ddx_idx]) <= self.U[i]
                    
                    self.model_dict[eval_t] +=   self.s[t][i+taker_ddx_idx]  <= self.U[i+3]
                    self.model_dict[eval_t] += -(self.s[t][i+taker_ddx_idx]) <= self.U[i+3]

        # Set relative constraints
        # Allowing STL predicates on relative distances, velocities, and accelerations
        #   Applied on s(t') for all t' in [-2, phi.horizon+1]
        for eval_t in range(0,self.phi.horizon): 
            for t in range(-2,self.phi.horizon+1):
                for i in [0,1,2]:
                    # relative position = giver position - taker position
                    self.model_dict[eval_t] += self.s[t][i+relative_x_idx] == self.s[t][i+giver_x_idx] - self.s[t][i+taker_x_idx]
                    # relative velocity = giver velocity - taker velocity 
                    self.model_dict[eval_t] += self.s[t][i+relative_dx_idx] == self.s[t][i+giver_dx_idx] - self.s[t][i+taker_dx_idx]
                    # relative acceleration = giver acceleration - taker acceleration 
                    self.model_dict[eval_t] += self.s[t][i+relative_ddx_idx] == self.s[t][i+giver_ddx_idx] - self.s[t][i+taker_ddx_idx]

        # Set Kshirsagars distance parameter
        #if self.KSHIRSAGAR_APPROACH:
            # K_p >= the l1 norm of relative distance
            # Constraints applied for all times during the handover, t' in [0, phi.horizon+1]
        delta = self.KSHIRSAGAR_DELTA # x,y,z offset of giver and taker hands on the object
            
        K_pluss = plp.LpVariable.dicts("K_pluss", (range(self.phi.horizon+1),range(3)), lowBound=0, cat=plp.LpContinuous)
        K_minus = plp.LpVariable.dicts("K_minus", (range(self.phi.horizon+1),range(3)), lowBound=0, cat=plp.LpContinuous)
        K_binary= plp.LpVariable.dicts("K_binary",(range(self.phi.horizon+1),range(3)), lowBound=0, cat=plp.LpBinary)
        for eval_t in range(0,self.phi.horizon): 
            for t in range(0,self.phi.horizon+1):
                for i in [0,1,2]:
                    self.model_dict[eval_t] += K_pluss[t][i] - K_minus[t][i] == (self.s[t][i+relative_x_idx] + delta[i]) # Giver - Taker + delta
                    self.model_dict[eval_t] += K_pluss[t][i] <= self.M * K_binary[t][i]
                    self.model_dict[eval_t] += K_minus[t][i] <= self.M * (1-K_binary[t][i])
                self.model_dict[eval_t] += self.s[t][K_p_idx] == K_pluss[t][0] + K_minus[t][0] + K_pluss[t][1] + K_minus[t][1] + K_pluss[t][2] + K_minus[t][2]
                                                                
        # Define Objective function
        #VELOCITY_WEIGHT = 1
        #objective_function = plp.LpAffineExpression()
        if self.OPTIMIZE_ROBUSTNESS:
            #objective_function = objective_function - rvar # Maximize (minimize negative) robustness
            for eval_t in range(0,self.phi.horizon):
                rvar = self.dict_vars['R_'+str(id(self.phi))+'_t0_'+str(0)+"_t_"+str(eval_t)]
                self.model_dict[eval_t] += -rvar
    
        #if self.MINIMIZE_AVG_VELOCITY:
        #    # minimize absolute velocity at all times
        #    abs_velocity = plp.LpVariable.dicts("abs_velocity",(range(self.phi.horizon+1),range(3)), lowBound=0, cat=plp.LpContinuous)
        #    for t in range(0,self.phi.horizon+1):
        #        for i in [0,1,2]:
        #            self.opt_model += abs_velocity[t][i] >= self.s[t][i+3]
        #            self.opt_model += abs_velocity[t][i] >= -self.s[t][i+3]
        #    objective_function = objective_function + plp.lpSum(abs_velocity)* VELOCITY_WEIGHT
    
        #if self.MINIMIZE_MAX_VELOCITY:
        #    # minimize maximum velocity for each axis
        #    max_abs_velocity = plp.LpVariable.dict("max_abs_velocity", (range(3)), lowBound=0, cat=plp.LpContinuous)
        #    for t in range(0,self.phi.horizon+1):
        #        for i in [0,1,2]:
        #            self.opt_model += max_abs_velocity[i] >= self.s[t][i+3]
        #            self.opt_model += max_abs_velocity[i] >= -self.s[t][i+3]
        #    objective_function = objective_function + plp.lpSum(max_abs_velocity) * VELOCITY_WEIGHT

        #recursive function
        USE_ONLY_UPPER_BOUND_ENCODING = True # Choose encoding using upper and lower bounds, or only upper bounds, relying on maximizing sense of the optimization to encode accurate robustness.
        def model_phi(t_0,phi,t,model_idx):
            if isinstance(phi, STLFormula.Predicate): # FLAGFLAGFLAG
                try:
                    rvar = self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar = plp.LpVariable('R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Continuous')
                    self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar
                if phi.operator == operatorclass.gt or  phi.operator == operatorclass.ge:
                    self.model_dict[model_idx] += self.s[t_0][phi.pi_index_signal] - phi.mu == rvar
                elif phi.operator == operatorclass.lt or phi.operator == operatorclass.le:
                    self.model_dict[model_idx] += -self.s[t_0][phi.pi_index_signal] + phi.mu == rvar
            
            elif isinstance(phi, STLFormula.TrueF):
                try:
                    rvar = self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar = plp.LpVariable('R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Continuous')
                    self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar
                self.model_dict[model_idx] += rvar >= self.M            
            
            elif isinstance(phi, STLFormula.FalseF):
                try:
                    rvar = self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar = plp.LpVariable('R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Continuous')
                    self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar
                self.model_dict[model_idx] += rvar <= -self.M
            
            elif isinstance(phi, STLFormula.Conjunction):
                model_phi(t_0,phi.first_formula,t,model_idx)
                model_phi(t_0,phi.second_formula,t,model_idx)
            
                try:
                    pvar1 = self.dict_vars['p_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    pvar1 = plp.LpVariable('p_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Binary')
                    self.dict_vars['p_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] = pvar1       
                try:
                    pvar2 = self.dict_vars['p_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    pvar2 = plp.LpVariable('p_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Binary')
                    self.dict_vars['p_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] = pvar2
                try:
                    rvar = self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar = plp.LpVariable('R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Continuous')
                    self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar
            
                self.model_dict[model_idx] += pvar1+pvar2 == 1 #(3)
                self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] #(4)
                self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] #(4)
                
                #self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] - (1 - pvar1)*self.M <= rvar <= self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar1)*self.M #(5)
                #self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] - (1 - pvar2)*self.M <= rvar <= self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar2)*self.M #(5)
                ### NEW BULLSHIT TEST
                if USE_ONLY_UPPER_BOUND_ENCODING == False:
                    self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] - (1 - pvar1)*self.M <= rvar #(5)
                    self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar1)*self.M #(5)
                    self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] - (1 - pvar2)*self.M <= rvar #(5)
                    self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar2)*self.M #(5)
                else:
                    self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar1)*self.M #(5)
                    self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar2)*self.M #(5)
            
            elif isinstance(phi, STLFormula.Disjunction):
                model_phi(t_0,phi.first_formula,t,model_idx)
                model_phi(t_0,phi.second_formula,t,model_idx)
            
                try:
                    pvar1 = self.dict_vars['p_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    pvar1 = plp.LpVariable('p_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Binary')
                    self.dict_vars['p_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] = pvar1       
                try:
                    pvar2 = self.dict_vars['p_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    pvar2 = plp.LpVariable('p_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Binary')
                    self.dict_vars['p_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] = pvar2
                try:
                    rvar = self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar = plp.LpVariable('R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Continuous')
                    self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar
            
                self.model_dict[model_idx] += pvar1+pvar2 == 1 #(3)
                self.model_dict[model_idx] += rvar >= self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] #(4)
                self.model_dict[model_idx] += rvar >= self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] #(4)
                
                #self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] - (1 - pvar1)*self.M <= rvar <= self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar1)*self.M #(5)
                #self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] - (1 - pvar2)*self.M <= rvar <= self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar2)*self.M #(5)
                # NEW BULLSHIT TEST
                if USE_ONLY_UPPER_BOUND_ENCODING == False:
                    self.model_dict[model_idx] += self.dict_vars['r_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] - (1 - pvar1)*self.m <= rvar #(5)
                    self.model_dict[model_idx] += rvar <= self.dict_vars['r_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar1)*self.m #(5)
                    self.model_dict[model_idx] += self.dict_vars['r_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] - (1 - pvar2)*self.m <= rvar #(5)
                    self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar2)*self.M #(5)
                else:
                    self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.first_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar1)*self.M #(5)
                    self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.second_formula))+'_t0_'+str(t_0)+'_t_'+str(t)] + (1 - pvar2)*self.M #(5)

            elif isinstance(phi,STLFormula.Always):
                try:
                    rvar = self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar = plp.LpVariable('R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Continuous')
                    self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar
                #aN_t = min(t+phi.t1, phi.horizon) # Raman definition
                #bN_t = min(t+phi.t2, phi.horizon)
                a = phi.t1
                b = phi.t2

                # My new definition
                N = self.phi.horizon
                atN_t0 = min( max(t_0+a, t+1), N) 
                btN_t0 = min( max(t_0+b, t),   N)

                # My old definition
                #atN_t0 = min( max(t_0+a, t+1), phi.horizon) 
                #btN_t0 = min( max(t_0+b, t),   phi.horizon)

                for t_i in range(atN_t0, btN_t0 + 1):
                    model_phi(t_i,phi.formula,t,model_idx)
                    
                    try:
                        pvar_i = self.dict_vars['p_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)]
                    except KeyError:
                        pvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t),cat='Binary')
                        self.dict_vars['p_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] = pvar_i
                    
                    self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] #(4)
                    
                    #self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] - (1 - pvar_i)*self.M <= rvar <= self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] + (1 - pvar_i)*self.M #(5)
                    ### NEW BULLSHIT TEST
                    if USE_ONLY_UPPER_BOUND_ENCODING == False:
                        self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] - (1 - pvar_i)*self.M <= rvar #(5)
                        self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] + (1 - pvar_i)*self.M #(5)
                    else:
                        self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] + (1 - pvar_i)*self.M #(5)

                ### NEW If the interval is empty, this condition is impossible!
                if atN_t0 < btN_t0:
                    self.model_dict[model_idx] += plp.lpSum([self.dict_vars['p_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] for t_i in range(atN_t0, btN_t0 +1)]) == 1 #(3)
            
            elif isinstance(phi,STLFormula.Eventually):
                try:
                    rvar = self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar = plp.LpVariable('R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Continuous')
                    self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar
                #aN_t = min(t+phi.t1, phi.horizon) # Raman definition
                #bN_t = min(t+phi.t2, phi.horizon)
                a = phi.t1
                b = phi.t2
                # My old definition
                #if t+1 < t_0 + b: 
                #    alphatN_t0 = t_0 + a
                #else:
                #    alphatN_t0 = t + 1
                #alphatN_t0 = min(alphatN_t0, phi.horizon)
                #btN_t0 = min( max(t_0+b, t), phi.horizon)
                
                # My new definition
                N = self.phi.horizon
                if t < t_0 + b:
                    alphatN_t0 = min(N, t_0 + a)
                else:
                    alphatN_t0 = min(N, t + 1)
                btN_t0 = min( max(t_0+b, t), N)
                
                for t_i in range(alphatN_t0,btN_t0+1):
                    model_phi(t_i,phi.formula,t,model_idx)
                
                    try:
                        pvar_i = self.dict_vars['p_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)]
                    except KeyError:
                        pvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t),cat='Binary')
                        self.dict_vars['p_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] = pvar_i
                    
                    self.model_dict[model_idx] += rvar >= self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] #(4)
                    
                    #self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] - (1 - pvar_i)*self.M <= rvar <= self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] + (1 - pvar_i)*self.M #(5)
                    ### NEW BULLSHIT TEST
                    if USE_ONLY_UPPER_BOUND_ENCODING == False:
                        self.model_dict[model_idx] += self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] - (1 - pvar_i)*self.M <= rvar #(5)
                        self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] + (1 - pvar_i)*self.M #(5)
                    else:
                        self.model_dict[model_idx] += rvar <= self.dict_vars['R_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] + (1 - pvar_i)*self.M #(5)

                ### NEW If the interval is empty, this condition is impossible! Only add constraint if interval exists!
                if alphatN_t0 < btN_t0:
                    self.model_dict[model_idx] += plp.lpSum([self.dict_vars['p_'+str(id(phi.formula))+'_t0_'+str(t_i)+'_t_'+str(t)] for t_i in range(alphatN_t0,btN_t0+1)]) == 1 #(3)
            
            elif isinstance(phi,STLFormula.Negation):
                try:
                    rvar = self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar = plp.LpVariable('R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Continuous')
                    self.dict_vars['R_'+str(id(phi))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar
                model_phi(t_0,phi.formula,t,model_idx)
                try:
                    rvar_i = self.dict_vars['p_'+str(id(phi.formula))+'_t0_'+str(t_0)+'_t_'+str(t)]
                except KeyError:
                    rvar_i = plp.LpVariable('p_'+str(id(phi.formula))+'_t0_'+str(t_0)+'_t_'+str(t),cat='Binary')
                    self.dict_vars['p_'+str(id(phi.formula))+'_t0_'+str(t_0)+'_t_'+str(t)] = rvar_i
                self.model_dict[model_idx] += rvar == -rvar_i
    
        for eval_t in range(0,self.phi.horizon):
            model_phi(0, self.phi , eval_t, model_idx=eval_t)
            rvar = self.dict_vars['R_'+str(id(self.phi))+'_t0_'+str(0)+'_t_'+str(eval_t)]
            if self.HARD_STL_CONSTRAINT:
                self.model_dict[eval_t] += rvar >= 0

            if self.KSHIRSAGAR_APPROACH:
                try:
                    Krvar =  self.dict_vars['R_'+str(id(self.Kshirsagar_phi))+'_t0_'+str(0)+'_t_'+str(eval_t)]
                except KeyError:
                    Krvar = plp.LpVariable('R_'+str(id(self.Kshirsagar_phi))+'_t0_'+str(0)+'_t_'+str(eval_t),cat='Continuous')
                    self.dict_vars['R_'+str(id(self.Kshirsagar_phi))+'_t0_'+str(0)+'_t_'+str(eval_t)] = Krvar
                    model_phi(0, self.Kshirsagar_phi, eval_t, model_idx=eval_t)
                if self.HARD_KSHIRSAGAR_CONSTRAINT:
                    self.model_dict[eval_t] += Krvar >= 0
        #else:
            #self.opt_model += rvar >= -1 # -epsilon
            #very_large_number = 100000000
            #negative_robustness = plp.LpVariable("negative_robustness", lowBound=0, cat=plp.LpContinuous)
            #self.opt_model += negative_robustness >= -rvar
            #objective_function = objective_function + negative_robustness* very_large_number
    
        ## Set objective function
        #self.opt_model += objective_function

        # build_model_time = time.monotonic() - build_model_start_time

    def get_giver_plan(self, n):
        """
            Returns position of the giver at time t_n
        """
        giver_pos_idx = [self.dimensions.index('giver_x'),self.dimensions.index('giver_y'),self.dimensions.index('giver_z')]
        return [self.s[n][idx].varValue for idx in giver_pos_idx]
        
    def get_taker_plan(self, n):
        """
            Returns position of the taker at time t_n
        """
        taker_pos_idx = [self.dimensions.index('taker_x'),self.dimensions.index('taker_y'),self.dimensions.index('taker_z')]
        return [self.s[n][idx].varValue for idx in taker_pos_idx]

    #def predict_human(self, role, n, prediction_len=1): # Outdated!!!!
    #    '''
    #        Predicts future position of human according to constant acceleration model
    #        Inputs:
    #            role: string "Giver" or "Taker", who is the target
    #            n: int, current step,start point of prediction
    #            prediction_len: int, how many steps are predicted
    #        Outputs:
                
    #    '''
    #    if role == "Taker":
    #        role_offset = 6
    #    elif role == "Giver":
    #        role_offset = 0
    #    else:
    #        raise

    #    predict_p = [ [0]*3 for dummy in range(prediction_len) ]

    #    for i in range(3):
    #        if n>1:
    #            acc = (self.s[n][i+role_offset].varValue + self.s[n-2][i+role_offset].varValue - 2*self.s[n-1][i+role_offset].varValue)
    #        else:
    #            acc = 0
    #        vel = (self.s[n][i+role_offset].varValue - self.s[n-1][i+role_offset].varValue)
    #        vel = vel + acc
    #        predict_p[0][i] = self.s[n][i+role_offset].varValue + vel
    #        for p in range(prediction_len-1):
    #            vel = vel + acc
    #            predict_p[p+1][i] = predict_p[p][i] + vel

    #    if self.phi.horizon > n+prediction_len:
    #        fix_t = list(range(n+1, n+prediction_len+1))
    #        fix_var = list(range(role_offset, role_offset+3))
    #        self.fix_variables(fix_t, fix_var, predict_p)
    #        return 1
    #    else:
    #        return 0



    def prediction_mse(self, true_giver, true_taker, t_start = 0, t_end = None):
        '''
            Calculate the Mean Square Error between recorded signal and predicted signal
            Input:
                true_giver: Recorded signal of giver, 2D np.array, true_giver[t_n][x/y/z]
                true_taker: Recorded signal of taker, 2D np.array, true_giver[t_n][x/y/z]
                t_start:    n for Start time, t_n, of signals for MSE
                t_end:      N for End time, t_N, of signals for MSE
            Output:
                self.mse_giver:  list, mean square error of  [x, y, z, euclidian distance]
                self.mse_taker:  scalar, mean square error   -||-

             1/(N-n) sum from n to N( (s_giver_x - true_giver_x)^2 )
        '''
        if t_end is None:
            t_end = self.phi.horizon

        #x_giver_prediction = [self.s[t][0].varValue for t in range(t_start, t_end+1)]
        #y_giver_prediction = [self.s[t][1].varValue for t in range(t_start, t_end+1)]
        #z_giver_prediction = [self.s[t][2].varValue for t in range(t_start, t_end+1)]
        
        #x_taker_prediction = [self.s[t][0+6].varValue for t in range(t_start, t_end+1)]
        #y_taker_prediction = [self.s[t][1+6].varValue for t in range(t_start, t_end+1)]
        #z_taker_prediction = [self.s[t][2+6].varValue for t in range(t_start, t_end+1)]

        self.mse_giver = [
            mse([self.s[t][0].varValue for t in range(t_start, t_end+1)], true_giver[t_start:t_end+1,0]),
            mse([self.s[t][1].varValue for t in range(t_start, t_end+1)], true_giver[t_start:t_end+1,1]),
            mse([self.s[t][2].varValue for t in range(t_start, t_end+1)], true_giver[t_start:t_end+1,2]),
            mse([[self.s[t][i].varValue for i in range(0,3)] for t in range(t_start, t_end+1)], true_giver[t_start:t_end+1]),
        ]
        self.mse_taker = [
            mse([self.s[t][0+6].varValue for t in range(t_start, t_end+1)], true_taker[t_start:t_end+1,0]),
            mse([self.s[t][1+6].varValue for t in range(t_start, t_end+1)], true_taker[t_start:t_end+1,1]),
            mse([self.s[t][2+6].varValue for t in range(t_start, t_end+1)], true_taker[t_start:t_end+1,2]),
            mse([[self.s[t][i+6].varValue for i in range(0,3)] for t in range(t_start, t_end+1)], true_taker[t_start:t_end+1]),
        ]

        return self.mse_giver, self.mse_taker

    

    
def plot_trajectory2(phi, current_step, trajectory, dt, robot_role = 'Giver', giver_mse = None, taker_mse = None, t_start_mse = None, save_flag = False, close_flag = True, title_txt = None , true_human_x=False, true_human_y=False, true_human_zG=False, true_human_zT=False):
    '''
        Plot trajectory without also plotting human data
    '''
    if t_start_mse is None:
        t_start_mse = current_step

    fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['legend',   'legend',   'legend']], layout='tight')
    ax_pos_x = ax_dict['x']
    ax_pos_y = ax_dict['y']
    ax_pos_zT = ax_dict['zT']
    ax_pos_zG = ax_dict['zG']
    ax_pos_x.grid(True)
    ax_pos_y.grid(True)
    ax_pos_zT.grid(True)
    ax_pos_zG.grid(True)

    #fig_pos = plt.figure()
    #ax_pos_x = fig_pos.add_subplot(131)
    #plt.grid(True)
    #ax_pos_y = fig_pos.add_subplot(132)
    #plt.grid(True)
    #ax_pos_z = fig_pos.add_subplot(133)
    #plt.grid(True)

    t_trajectory = np.linspace(0, phi.horizon*dt, phi.horizon+1)
    #t_human = list(human_data.time)
    N = phi.horizon
        
    x_giver = [x[0] for x in trajectory]
    y_giver = [x[1] for x in trajectory]
    z_giver = [x[2] for x in trajectory]
    x_taker = [x[6] for x in trajectory]
    y_taker = [x[7] for x in trajectory]
    z_taker = [x[8] for x in trajectory]
    
    # x
    ax_pos_x.plot(t_trajectory[0:current_step+1], x_giver[0:current_step+1], '-g', marker='o', label=r'Giver past')
    ax_pos_x.plot(t_trajectory[current_step+1:N], x_giver[current_step+1:N], '-g', marker='x', label=r'Giver ')        
    
    ax_pos_x.plot(t_trajectory[0:current_step+1], x_taker[0:current_step+1], '-r', marker='o', label=r'Taker past')
    ax_pos_x.plot(t_trajectory[current_step+1:N], x_taker[current_step+1:N], '-r', marker='x', label=r'Taker ')

    if true_human_x:
        ax_pos_x.plot(t_trajectory, true_human_x, '-m', alpha=0.5, label=r'human')
        #ax_pos_x.plot(t_human, human_data.x_taker_RHand, '-b', alpha=0.5, label=r'human taker')

    ax_pos_x.set_xlabel('t')
    ax_pos_x.set_ylabel('X')
    
    # y
    ax_pos_y.plot(t_trajectory[0:current_step+1], y_giver[0:current_step+1], '-g', marker='o', label=r'Giver known')
    ax_pos_y.plot(t_trajectory[current_step+1:N], y_giver[current_step+1:N], '-g', marker='x', label=r'Giver prediction')
    ax_pos_y.plot(t_trajectory[0:current_step+1], y_taker[0:current_step+1], '-r', marker='o', label=r'Taker known')
    ax_pos_y.plot(t_trajectory[current_step+1:N], y_taker[current_step+1:N], '-r', marker='x', label=r'Taker prediction')
      
    if true_human_y:
        ax_pos_y.plot(t_trajectory, true_human_y, '-m', alpha=0.5, label=r'human')
    #ax_pos_y.plot(t_human, human_data.y_giver_RHand, '-m', alpha=0.5, label=r'human giver')
    #ax_pos_y.plot(t_human, human_data.y_taker_RHand, '-b', alpha=0.5, label=r'human taker')
    
    ax_pos_y.set_xlabel('t')
    ax_pos_y.set_ylabel('Y')
    # z
    ax_pos_zG.plot(t_trajectory[0:current_step+1], z_giver[0:current_step+1], '-g', marker='o', label=r'Giver past steps')
    ax_pos_zG.plot(t_trajectory[current_step+1:N], z_giver[current_step+1:N], '-g', marker='x', label=r'Giver controll sequence')        

    ax_pos_zT.plot(t_trajectory[current_step+1:N], z_taker[current_step+1:N], '-r', marker='x', label=r'Taker prediction')
    ax_pos_zT.plot(t_trajectory[0:current_step+1], z_taker[0:current_step+1], '-r', marker='o', label=r'Taker observation')
    
    if true_human_zG:
        ax_pos_zG.plot(t_trajectory, true_human_zG, '-m', alpha=0.5, label=r'human')
    if true_human_zT:
        ax_pos_zT.plot(t_trajectory, true_human_zT, '-m', alpha=0.5, label=r'human')
    #ax_pos_zG.plot(t_human, human_data.z_giver_RHand, '-m', alpha=0.5, label=r'human giver')
    #ax_pos_zT.plot(t_human, human_data.z_taker_RHand, '-b', alpha=0.5, label=r'human taker')
    
    ax_pos_zG.set_xlabel('t')
    ax_pos_zG.set_ylabel('Z')
    ax_pos_zT.set_xlabel('t')
    ax_pos_zT.set_ylabel('Z')

    # Title
    robust_str = str(round(phi.robustness(trajectory,0),3))
    if robot_role == 'Giver':
        role_text = r'Robot giver, Human taker \\'
    elif robot_role == 'Taker':
        role_text = r'Human giver, Robot taker \\'

    if title_txt is None:
        ax_pos_y.set_title( role_text + r'quantitave $\rho =' + robust_str + r' , t = t_{' + str(current_step) + r'}$')
    else:
        ax_pos_y.set_title(title_txt + r'\\' + role_text + r'quantitave $\rho =' + robust_str + r' , t = t_{' + str(current_step) + r'}$')

    # MSE
    if not( giver_mse is None ) and not( taker_mse is None ):
        fig_pos.text(0.98, 0.98, # 'Giver mse:\n $x={0:.3f}$\n $y={1:.3f}$\n $z={2:.3f}$\n $xyz={3:.3f}$'.format(1,2,3,4))
                "Giver mse: $x = {:.4f}, y = {:.4f}, z = {:.4f}, [x,y,z]^T = {:.4f}$\n".format(giver_mse[0],giver_mse[1],giver_mse[2],giver_mse[3])
                + 'Taker mse: $x = {:.4f}, y = {:.4f}, z = {:.4f}, [x,y,z]^T = {:.4f}$'.format(taker_mse[0],taker_mse[1],taker_mse[2],taker_mse[3]),
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(boxstyle="round",facecolor='white',edgecolor='black')
        )

    # Legend
    ax_dict['legend'].set_visible(False)
    ax_dict['legend'].set_box_aspect(0.001)
    handles, labels = ax_pos_y.get_legend_handles_labels()
    fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=6)

    #ax_pos_y.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=3)
    #ax_dict['legend'].legend(handles, labels, loc='lower center' ,shadow=True, ncol=6)

    # Axis limits
    ax_pos_x.set_ylim(-0.6, 0.6)
    ax_pos_y.set_ylim(-0.4, 0.4)
    ax_pos_zT.set_ylim(-0.4, 0.1)
    ax_pos_zG.set_ylim(-0.4, 0.1)

    ax_pos_x.set_xlim(-0.2, 3.2)
    ax_pos_y.set_xlim(-0.2, 3.2)
    ax_pos_zT.set_xlim(-0.2, 3.2)
    ax_pos_zG.set_xlim(-0.2, 3.2)

    fig_pos.set_figheight(6)
    fig_pos.set_figwidth(12)

    #ax_pos_x.figure.subplotpars.bottom = 0.2
    #fig_pos.subplots(figsize=(4,2))

    #fig_pos.tight_layout()
    
    #plt.show()
    
    if save_flag:
        plt.savefig(PLOT_PATH + PLOT_NAME + f't{current_step:02}.png')

    if close_flag:
        plt.close(fig_pos)

def plot_trajectory(phi, current_step, trajectory, human_data, dt, giver_mse = None, taker_mse = None, t_start_mse = None, save_flag = False, close_flag = True, title_txt = None ):

    if t_start_mse is None:
        t_start_mse = current_step

    fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['legend',   'legend',   'legend']], layout='tight')
    ax_pos_x = ax_dict['x']
    ax_pos_y = ax_dict['y']
    ax_pos_zT = ax_dict['zT']
    ax_pos_zG = ax_dict['zG']
    ax_pos_x.grid(True)
    ax_pos_y.grid(True)
    ax_pos_zT.grid(True)
    ax_pos_zG.grid(True)

    #fig_pos = plt.figure()
    #ax_pos_x = fig_pos.add_subplot(131)
    #plt.grid(True)
    #ax_pos_y = fig_pos.add_subplot(132)
    #plt.grid(True)
    #ax_pos_z = fig_pos.add_subplot(133)
    #plt.grid(True)

    t_trajectory = np.linspace(0, phi.horizon*dt, phi.horizon+1)
    t_human = list(human_data.time)
    N = phi.horizon
        
    x_giver = [x[0] for x in trajectory]
    y_giver = [x[1] for x in trajectory]
    z_giver = [x[2] for x in trajectory]
    x_taker = [x[6] for x in trajectory]
    y_taker = [x[7] for x in trajectory]
    z_taker = [x[8] for x in trajectory]
    
    # x
    ax_pos_x.plot(t_trajectory[0:current_step+1], x_giver[0:current_step+1], '-g', marker='o', label=r'Giver past steps')
    ax_pos_x.plot(t_trajectory[current_step+1:N], x_giver[current_step+1:N], '-g', marker='x', label=r'Giver controll sequence')        
    
    ax_pos_x.plot(t_trajectory[0:current_step+1], x_taker[0:current_step+1], '-r', marker='o', label=r'Taker observation')
    ax_pos_x.plot(t_trajectory[current_step+1:N], x_taker[current_step+1:N], '-r', marker='x', label=r'Taker prediction')

    ax_pos_x.plot(t_human, human_data.x_giver_RHand, '-m', alpha=0.5, label=r'human giver')
    ax_pos_x.plot(t_human, human_data.x_taker_RHand, '-b', alpha=0.5, label=r'human taker')

    ax_pos_x.set_xlabel('t')
    ax_pos_x.set_ylabel('X')
    
    # y
    ax_pos_y.plot(t_trajectory[0:current_step+1], y_giver[0:current_step+1], '-g', marker='o', label=r'Giver known')
    ax_pos_y.plot(t_trajectory[current_step+1:N], y_giver[current_step+1:N], '-g', marker='x', label=r'Giver prediction')
    ax_pos_y.plot(t_trajectory[0:current_step+1], y_taker[0:current_step+1], '-r', marker='o', label=r'Taker known')
    ax_pos_y.plot(t_trajectory[current_step+1:N], y_taker[current_step+1:N], '-r', marker='x', label=r'Taker prediction')
         
    ax_pos_y.plot(t_human, human_data.y_giver_RHand, '-m', alpha=0.5, label=r'human giver')
    ax_pos_y.plot(t_human, human_data.y_taker_RHand, '-b', alpha=0.5, label=r'human taker')
    
    ax_pos_y.set_xlabel('t')
    ax_pos_y.set_ylabel('Y')
    # z
    ax_pos_zG.plot(t_trajectory[0:current_step+1], z_giver[0:current_step+1], '-g', marker='o', label=r'Giver past steps')
    ax_pos_zG.plot(t_trajectory[current_step+1:N], z_giver[current_step+1:N], '-g', marker='x', label=r'Giver controll sequence')        

    ax_pos_zT.plot(t_trajectory[current_step+1:N], z_taker[current_step+1:N], '-r', marker='x', label=r'Taker prediction')
    ax_pos_zT.plot(t_trajectory[0:current_step+1], z_taker[0:current_step+1], '-r', marker='o', label=r'Taker observation')

    ax_pos_zG.plot(t_human, human_data.z_giver_RHand, '-m', alpha=0.5, label=r'human giver')
    ax_pos_zT.plot(t_human, human_data.z_taker_RHand, '-b', alpha=0.5, label=r'human taker')
    
    ax_pos_zG.set_xlabel('t')
    ax_pos_zG.set_ylabel('Z')
    ax_pos_zT.set_xlabel('t')
    ax_pos_zT.set_ylabel('Z')

    # Title
    robust_str = str(round(phi.robustness(trajectory,0),3))
    if title_txt is None:
        ax_pos_y.set_title(r'Robot giver, Human taker \\' + r'quantitave $\rho =' + robust_str + r' , t = t_{' + str(current_step) + r'}$')
    else:
        ax_pos_y.set_title(title_txt + r'\\Robot giver, Human taker \\' + r'quantitave $\rho =' + robust_str + r' , t = t_{' + str(current_step) + r'}$')

    # MSE
    if not( giver_mse is None ) and not( taker_mse is None ):
        fig_pos.text(0.98, 0.98, # 'Giver mse:\n $x={0:.3f}$\n $y={1:.3f}$\n $z={2:.3f}$\n $xyz={3:.3f}$'.format(1,2,3,4))
                "Giver mse: $x = {:.4f}, y = {:.4f}, z = {:.4f}, [x,y,z]^T = {:.4f}$\n".format(giver_mse[0],giver_mse[1],giver_mse[2],giver_mse[3])
                + 'Taker mse: $x = {:.4f}, y = {:.4f}, z = {:.4f}, [x,y,z]^T = {:.4f}$'.format(taker_mse[0],taker_mse[1],taker_mse[2],taker_mse[3]),
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(boxstyle="round",facecolor='white',edgecolor='black')
        )

    # Legend
    ax_dict['legend'].set_visible(False)
    ax_dict['legend'].set_box_aspect(0.001)
    handles, labels = ax_pos_y.get_legend_handles_labels()
    fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=6)

    #ax_pos_y.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=3)
    #ax_dict['legend'].legend(handles, labels, loc='lower center' ,shadow=True, ncol=6)

    # Axis limits
    ax_pos_x.set_ylim(-0.6, 0.6)
    ax_pos_y.set_ylim(-0.4, 0.4)
    ax_pos_zT.set_ylim(-0.4, 0.1)
    ax_pos_zG.set_ylim(-0.4, 0.1)

    ax_pos_x.set_xlim(-0.2, 3.2)
    ax_pos_y.set_xlim(-0.2, 3.2)
    ax_pos_zT.set_xlim(-0.2, 3.2)
    ax_pos_zG.set_xlim(-0.2, 3.2)

    fig_pos.set_figheight(6)
    fig_pos.set_figwidth(12)

    #ax_pos_x.figure.subplotpars.bottom = 0.2
    #fig_pos.subplots(figsize=(4,2))

    #fig_pos.tight_layout()
    
    #plt.show()
    
    if save_flag:
        plt.savefig(PLOT_PATH + PLOT_NAME + f't{current_step:02}.png')

    if close_flag:
        plt.close(fig_pos)


def plot_in_worldframe(x1, y1, z1, x2, y2, z2, chest_Giver, chest_Taker, dt=0.2, current_step=0 , true_human_x=False, true_human_y=False, true_human_zG=False, true_human_zT=False):
    """
        Plot in worldframe, a trajectory given in shared frame
    """
    x1_world = np.zeros(len(x1))
    y1_world = np.zeros(len(x1))
    z1_world = np.zeros(len(x1))
    x2_world = np.zeros(len(x1))
    y2_world = np.zeros(len(x1))
    z2_world = np.zeros(len(x1))

    # Transform
    for t in range(len(x1)):
        TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
        
        P_n = np.matmul(TF, np.array([x1[t], y1[t], z1[t], 1]))  #giver point at time n
        x1_world[t] = P_n[0]
        y1_world[t] = P_n[1]
        z1_world[t] = P_n[2] # TF to shared frame

        P_n = np.matmul(TF, np.array([x2[t], y2[t], z2[t], 1]))  #taker point at time n
        x2_world[t] = P_n[0]
        y2_world[t] = P_n[1]
        z2_world[t] = P_n[2] # TF to shared frame

        if true_human_x:
            if true_human_zG:
                P_n = np.matmul(TF, np.array([true_human_x[t], true_human_y[t], true_human_zG[t], 1]))  #human point at time n
            if true_human_zT:
                P_n = np.matmul(TF, np.array([true_human_x[t], true_human_y[t], true_human_zT[t], 1]))  #human point at time n
            
            true_human_x[t] = P_n[0]
            true_human_y[t] = P_n[1]
            if true_human_zG:
                true_human_zG[t] = P_n[2]
            if true_human_zT:
                true_human_zT[t] = P_n[2]

        #if t_start_mse is None:
        #    t_start_mse = current_step
        

    # plot the figure
    fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zT'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['x',        'y',        'zG'     ],
                                           ['legend',   'legend',   'legend']], layout='tight')
    ax_pos_x = ax_dict['x']
    ax_pos_y = ax_dict['y']
    ax_pos_z1 = ax_dict['zG']
    ax_pos_z2 = ax_dict['zT']
    ax_pos_x.grid(True)
    ax_pos_y.grid(True)
    ax_pos_z1.grid(True)
    ax_pos_z2.grid(True)

    #fig_pos = plt.figure()
    #ax_pos_x = fig_pos.add_subplot(131)
    #plt.grid(True)
    #ax_pos_y = fig_pos.add_subplot(132)
    #plt.grid(True)
    #ax_pos_z = fig_pos.add_subplot(133)
    #plt.grid(True)

    N = len(x1_world)
    t_trajectory = np.linspace(0, N*dt, N)
    t_human = np.linspace(0, len(true_human_x)*dt, len(true_human_x))
        
    #current_step = N

    x_giver = x1_world
    y_giver = y1_world
    z_giver = z1_world
    x_taker = x2_world
    y_taker = y2_world
    z_taker = z2_world
    
    # x
    ax_pos_x.plot(t_trajectory[0:current_step+1], x_giver[0:current_step+1], '-g', marker='o', label=r'Giver past')
    ax_pos_x.plot(t_trajectory[current_step+1:N], x_giver[current_step+1:N], '-g', marker='x', label=r'Giver ')        
    
    ax_pos_x.plot(t_trajectory[0:current_step+1], x_taker[0:current_step+1], '-r', marker='o', label=r'Taker past')
    ax_pos_x.plot(t_trajectory[current_step+1:N], x_taker[current_step+1:N], '-r', marker='x', label=r'Taker ')

    if true_human_x:
        ax_pos_x.plot(t_trajectory, true_human_x, '-m', alpha=0.5, label=r'human')
        #ax_pos_x.plot(t_human, human_data.x_taker_RHand, '-b', alpha=0.5, label=r'human taker')

    ax_pos_x.set_xlabel('t')
    ax_pos_x.set_ylabel('X')
    
    # y
    ax_pos_y.plot(t_trajectory[0:current_step+1], y_giver[0:current_step+1], '-g', marker='o', label=r'Giver known')
    ax_pos_y.plot(t_trajectory[current_step+1:N], y_giver[current_step+1:N], '-g', marker='x', label=r'Giver prediction')
    ax_pos_y.plot(t_trajectory[0:current_step+1], y_taker[0:current_step+1], '-r', marker='o', label=r'Taker known')
    ax_pos_y.plot(t_trajectory[current_step+1:N], y_taker[current_step+1:N], '-r', marker='x', label=r'Taker prediction')
      
    if true_human_y:
        ax_pos_y.plot(t_trajectory, true_human_y, '-m', alpha=0.5, label=r'human')
    #ax_pos_y.plot(t_human, human_data.y_giver_RHand, '-m', alpha=0.5, label=r'human giver')
    #ax_pos_y.plot(t_human, human_data.y_taker_RHand, '-b', alpha=0.5, label=r'human taker')
    
    ax_pos_y.set_xlabel('t')
    ax_pos_y.set_ylabel('Y')
    # z
    ax_pos_z1.plot(t_trajectory[0:current_step+1], z_giver[0:current_step+1], '-g', marker='o', label=r'Giver past steps')
    ax_pos_z1.plot(t_trajectory[current_step+1:N], z_giver[current_step+1:N], '-g', marker='x', label=r'Giver controll sequence')        

    ax_pos_z2.plot(t_trajectory[current_step+1:N], z_taker[current_step+1:N], '-r', marker='x', label=r'Taker prediction')
    ax_pos_z2.plot(t_trajectory[0:current_step+1], z_taker[0:current_step+1], '-r', marker='o', label=r'Taker observation')
    
    if true_human_zG:
        ax_pos_z1.plot(t_trajectory, true_human_zG, '-m', alpha=0.5, label=r'human')
    if true_human_zT:
        ax_pos_z2.plot(t_trajectory, true_human_zT, '-m', alpha=0.5, label=r'human')
    #ax_pos_zG.plot(t_human, human_data.z_giver_RHand, '-m', alpha=0.5, label=r'human giver')
    #ax_pos_zT.plot(t_human, human_data.z_taker_RHand, '-b', alpha=0.5, label=r'human taker')
    
    ax_pos_z1.set_xlabel('t')
    ax_pos_z1.set_ylabel('Z')
    ax_pos_z2.set_xlabel('t')
    ax_pos_z2.set_ylabel('Z')

    # Title
    #robust_str = str(round(phi.robustness(trajectory,0),3))
    #if robot_role == 'Giver':
    #    role_text = r'Robot giver, Human taker \\'
    #elif robot_role == 'Taker':
    #    role_text = r'Human giver, Robot taker \\'

    #if title_txt is None:
    #    ax_pos_y.set_title( role_text + r'quantitave $\rho =' + robust_str + r' , t = t_{' + str(current_step) + r'}$')
    #else:
    ax_pos_y.set_title(r'Handover trajectory in world frame\\ $t = t_{' + str(current_step) + r'}$')

    # Legend
    ax_dict['legend'].set_visible(False)
    ax_dict['legend'].set_box_aspect(0.001)
    handles, labels = ax_pos_y.get_legend_handles_labels()
    fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=6)

    #ax_pos_y.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=3)
    #ax_dict['legend'].legend(handles, labels, loc='lower center' ,shadow=True, ncol=6)

    # Axis limits
    #ax_pos_x.set_ylim(-0.6, 0.6)
    #ax_pos_y.set_ylim(-0.4, 0.4)
    #ax_pos_z1.set_ylim(-0.4, 0.1)
    #ax_pos_z2.set_ylim(-0.4, 0.1)

    #ax_pos_x.set_xlim(-0.2, 3.2)
    #ax_pos_y.set_xlim(-0.2, 3.2)
    #ax_pos_z1.set_xlim(-0.2, 3.2)
    #ax_pos_z2.set_xlim(-0.2, 3.2)

    fig_pos.set_figheight(6)
    fig_pos.set_figwidth(12)

    #ax_pos_x.figure.subplotpars.bottom = 0.2
    #fig_pos.subplots(figsize=(4,2))

    #fig_pos.tight_layout()

    #if close_flag:
    #    plt.close(fig_pos)



































 # Demo
if __name__ == '__main__':
    import pandas as pd
    from math import floor
    mpl.rcParams['text.usetex'] = True

    

    KSHIRSAGAR_APPROACH = True

    OPTIMIZE_ROBUSTNESS=True
    QUANTITATIVE_OPTIMIZATION= True
    HARD_STL_CONSTRAINT= False
    
    # Output path for plots
    PLOT_PATH = 'G:/My Drive/Exjobb Drive folder/Exjobb/STL Generate Trajectory MILP/active-learn-stl-master/active-learn-stl-master/Plot_animation_1'
    PLOT_NAME = f'/06_RobotGiver_HumanTaker_OptRho{OPTIMIZE_ROBUSTNESS:01}_QuOpt{QUANTITATIVE_OPTIMIZATION:01}_HardSTL{HARD_STL_CONSTRAINT:01}_'

    # Choose plots
    PLOT_POSITION =     True
    
    # Add recorded data to all plots
    PLOT_HUMAN_GIVER =  True
    PLOT_HUMAN_TAKER =  True

    #CONSTANTS
    INDEX_X = 0 # Position
    INDEX_Y = 1 # Position
    INDEX_Z = 2 # Position
    INDEX_dX = 3 # Speed
    INDEX_dY = 4 # Speed
    INDEX_dZ = 5 # Speed
    INDEX_XX = 6 # Relative position
    INDEX_YY = 7 # Relative position
    INDEX_ZZ = 8 # Relative position
    dimensions = ['giver_x','giver_y','giver_z','giver_dx','giver_dy','giver_dz', 'taker_x', 'taker_y', 'taker_z', 'taker_dx', 'taker_dy', 'taker_dz', 'relative_x','relative_y','relative_z','K_p']
    dt = 0.1 # [s] step time

    # Load recorded human handover
    human_data = pd.read_csv('Human_handover_signal\P1_S1_validation_handover_5.csv')
    # access with   "human_data.[dimension]_[giv-/taker]_RHand[::DOWNSAMPLING]", ex human_data.dx_giver_RHand
    DOWNSAMPLING = 1
    #dt = human_data.time[DOWNSAMPLING]-human_data.time[0]
    print("dt = "+str(dt))

    # List of STL parameters on the form
    #   Psi_[t1, t2] (lb < var < ub)
    # Demo list
    STL_list = [['F', 0.814172763687,   0.909099217176, -0.350859388685,    'giver_x',    -0.207550246362],
                ['G', 0.758981954831,   1.40358411718,  -0.261914436996,    'giver_y',    -0.0364622240287],
                ['G', 0.35376972241,    0.58113637733,  -0.395706008222,    'giver_z',    -0.117180705801],
                ['G', 0.5,              1.0,            -0.4,               'giver_dx',   0.4],
                ['F', 0.8,              1.0,            -0.5,               'relative_x',   0.1]]
    # Position only
    #STL_list = [['F', 0.814172763687,   0.909099217176, -0.350859388685,    'giver_x',    -0.207550246362],
    #            ['F', 0.382622750271,   0.646810665587, -0.390333086401,    'giver_x',    -0.276653912559],
    #            ['F', 1.26102279934,    1.46198075017,  -0.246882783882,    'giver_x',    -0.126925546954],
    #            ['F', 1.57254764879,    1.91666698685,  -0.222190309584,    'giver_x',    -0.103794275856],
    #            ['F', 2.93913036107,    3.41679378433,  -0.499519526182,    'giver_x',    -0.378951790397],
    #            ['F', 0.701108019629,   0.934202288784, -0.255286278676,    'giver_y',    -0.13580185447],
    #            ['F', 1.45638857223,    1.5833579677,   -0.144614700758,    'giver_y',    -0.0302115290624],
    #            ['F', 3.06716021847,    3.18068944681,  -0.242035295306,    'giver_y',    -0.114240444213],
    #            ['F', 0.333168563053,   0.727071080009, -0.343475869826,    'giver_z',    -0.18929078807],
    #            ['F', 0.623563213883,   1.16853044839,  -0.227254367136,    'giver_z',    -0.101338857201],
    #            ['F', 1.08333336669,    1.91959785164,  -0.157971158313,    'giver_z',    -0.0398491602237],
    #            ['F', 2.5,              3.24532016581,  -0.38958643352,     'giver_z',    -0.275912744875],
    #            ['G', 0.156849319234,   0.577801427841, -0.438828699264,    'giver_x',    -0.281227611555],
    #            ['G', 0.761272060317,   0.931506571637, -0.355946525461,    'giver_x',    -0.191624882714],
    #            ['G', 1.00740626808,    1.29728312259,  -0.313303367031,    'giver_x',    -0.1347773075],
    #            ['G', 1.44787494342,    1.65687611785,  -0.243941446864,    'giver_x',    -0.10802152059],
    #            ['G', 0.0402097227964,  0.560290654603, -0.319794077243,    'giver_y',    -0.187513799401],
    #            ['G', 0.758981954831,   1.40358411718,  -0.261914436996,    'giver_y',    -0.0364622240287],
    #            ['G', 1.37071183342,    1.78056559322,  -0.152330935085,    'giver_y',    -0.0113440840476],
    #            ['G', 1.77760304876,    1.95276597941,  -0.12851523791,     'giver_y',    -0.00246073715382],
    #            ['G', 0.35376972241,    0.58113637733,  -0.395706008222,    'giver_z',    -0.117180705801],
    #            ['G', 0.250630056992,   0.330920682341, -0.406399835199,    'giver_z',    -0.210672255096],
    #            ['G', 0.715850860261,   0.874999988468, -0.309027108745,    'giver_z',    -0.0402592000701],
    #            ['G', 1.41948036745,    1.70201463388,  -0.16663890725,     'giver_z',    0.0132424168066],
    #            ['G', 1.11143543532,    1.40706244707,  -0.212062909607,    'giver_z',    0.0020066321825],
    #            ['G', 3.08333414037,    3.19878305137,  -0.441910714095,    'giver_z',    -0.277924518036]
    #            ]

    # Position, Speed and Relative position
    STL_list = [
                ['F', 0.814172763687,   0.909099217176, -0.350859388685,    'giver_x',    -0.207550246362], #pos
                ['F', 0.382622750271,   0.646810665587, -0.390333086401,    'giver_x',    -0.276653912559],
                ['F', 1.26102279934,    1.46198075017,  -0.246882783882,    'giver_x',    -0.126925546954],
                ['F', 1.57254764879,    1.91666698685,  -0.222190309584,    'giver_x',    -0.103794275856],
                #['F', 2.93913036107,    3.41679378433,  -0.499519526182,    'giver_x',    -0.378951790397],
                ['F', 0.701108019629,   0.934202288784, -0.255286278676,    'giver_y',    -0.13580185447],
                ['F', 1.45638857223,    1.5833579677,   -0.144614700758,    'giver_y',    -0.0302115290624],
                #['F', 3.06716021847,    3.18068944681,  -0.242035295306,    'giver_y',    -0.114240444213],
                ['F', 0.333168563053,   0.727071080009, -0.343475869826,    'giver_z',    -0.18929078807],
                ['F', 0.623563213883,   1.16853044839,  -0.227254367136,    'giver_z',    -0.101338857201],
                ['F', 1.08333336669,    1.91959785164,  -0.157971158313,    'giver_z',    -0.0398491602237],
                #['F', 2.5,              3.24532016581,  -0.38958643352,     'giver_z',    -0.275912744875],
                ['G', 0.156849319234,   0.577801427841, -0.438828699264,    'giver_x',    -0.281227611555],
                ['G', 0.761272060317,   0.931506571637, -0.355946525461,    'giver_x',    -0.191624882714],
                ['G', 1.00740626808,    1.29728312259,  -0.313303367031,    'giver_x',    -0.1347773075],
                ['G', 1.44787494342,    1.65687611785,  -0.243941446864,    'giver_x',    -0.10802152059],
                ['G', 0.0402097227964,  0.560290654603, -0.319794077243,    'giver_y',    -0.187513799401],
                ['G', 0.758981954831,   1.40358411718,  -0.261914436996,    'giver_y',    -0.0364622240287],
                ['G', 1.37071183342,    1.78056559322,  -0.152330935085,    'giver_y',    -0.0113440840476],
                ['G', 1.77760304876,    1.95276597941,  -0.12851523791,     'giver_y',    -0.00246073715382],
                ['G', 0.35376972241,    0.58113637733,  -0.395706008222,    'giver_z',    -0.117180705801],
                ['G', 0.250630056992,   0.330920682341, -0.406399835199,    'giver_z',    -0.210672255096],
                ['G', 0.715850860261,   0.874999988468, -0.309027108745,    'giver_z',    -0.0402592000701],
                ['G', 1.41948036745,    1.70201463388,  -0.16663890725,     'giver_z',    0.0132424168066],
                ['G', 1.11143543532,    1.40706244707,  -0.212062909607,    'giver_z',    0.0020066321825],
                #['G', 3.08333414037,    3.19878305137,  -0.441910714095,    'giver_z',    -0.277924518036],
                ['F', 0.995881063376,   1.18315784202,  0.132044435283,     'giver_dx',   0.253670657795], #spd
                ['F', 1.61606710323,    2.26339904769,  0.00873653553122,   'giver_dx',   0.123594488542],
                #['F', 3.07593291741,    3.46200383214,  -0.0801109000666,   'giver_dx',   0.0392187920477],
                ['F', 0.547292722055,   0.836312820878, 0.0967082355789,    'giver_dy',   0.209545401265],
                ['F', 0.932875170217,   1.10944770426,  0.143570845747,     'giver_dy',   0.262838341005],
                ['F', 1.47104521542,    1.93353333165,  0.00191985136202,   'giver_dy',   0.113528072205],
                #['F', 2.77749998103,    3.0,            -0.170933287468,    'giver_dy',   0.077613031405],
                ['F', 0.5,              0.813304577082, 0.224494839382,     'giver_dz',   0.339657273664],
                ['F', 1.04108739178,    1.95891071324,  0.0114944136338,    'giver_dz',   0.12315111132],
                #['F', 2.5,              3.41779559705,  -0.064026674879,    'giver_dz',   0.0477439791184],
                ['G', 0.286324978527,   0.665447678226, -0.0439740432305,   'giver_dx',   0.33133146286],
                ['G', 0.920795969231,   1.15538185521,  0.0470446368607,    'giver_dx',   0.305470674212],
                ['G', 1.18168917329,    1.37475082571,  0.0272519296249,    'giver_dx',   0.260918602631],
                ['G', 1.79176459585,    2.32069082768,  -1.0,               'giver_dx',   0.219724591701],
                #['G', 2.84097801719,    3.28953697047,  -0.492401740874,    'giver_dx',   0.387228876622],
                ['G', 0.422890230049,   0.566233691377, -0.00777631968056,  'giver_dy',   0.173484100909],
                ['G', 0.963912347688,   1.20805478673,  0.123689738539,     'giver_dy',   0.283867552143],
                ['G', 1.94263504624,    2.28233903326,  -0.580448420103,    'giver_dy',   0.140078142668],
                #['G', 2.47728088873,    2.61980168125,  -0.288952161911,    'giver_dy',   0.223684864415],
                #['G', 2.71752947504,    2.93219549799,  -0.253872091649,    'giver_dy',   0.153798175683],
                ['G', 0.723083981301,   0.848548628638, 0.176532147077,     'giver_dz',   0.369826771858],
                ['G', 0.381592925361,   0.497129294584, 0.0986092013991,    'giver_dz',   0.339207041745],
                ['G', 1.12591002543,    1.54028152882,  -0.0648703470417,   'giver_dz',   0.248717654734],
                #['G', 2.3483659708,     2.89635007603,  -0.905123757408,    'giver_dz',   0.174556690252],
                ['F', 0.45151992067,    0.958333313541, -0.718921222178,    'relative_x',   -0.606530518056], # x_r - x_h
                ['F', 0.814012841836,   1.2631619009,   -0.487864139735,    'relative_x',   -0.359031380041],
                ['F', 1.14056544985,    1.79166680568,  -0.38146535084,     'relative_x',   -0.271534421066],
                ['F', 1.70833336583,    2.17630189877,  -0.391132256047,    'relative_x',   -0.277722034468],
                #['F', 2.93812556505,    3.46990797439,  -0.896678913883,    'relative_x',   -0.777997491135],
                ['F', 0.257858527116,   0.958335166487, -0.406718646594,    'relative_y',   -0.290730908156],
                ['F', 1.07700275154,    1.75032203609,  -0.197414658901,    'relative_y',   -0.0823084067944],
                #['F', 2.744773772,      3.49309123271,  -0.515503606906,    'relative_y',   -0.379300242769],
                ['F', 0.0,              0.419296441916, -0.118960037487,    'relative_z',   -0.000572162328359],
                ['F', 0.670841131923,   0.934703011289, -0.0806280787225,   'relative_z',   0.0509139326776],
                ['F', 1.37141936279,    1.5945789738,   -0.0628157904462,   'relative_z',   0.053566156175],
                ['F', 1.70833332672,    2.41673736958,  -0.0777643439978,   'relative_z',   0.03624460266],
                #['F', 2.41666503126,    2.87718024629,  -0.153965891342,    'relative_z',   -0.0364945462539],
                #['F', 3.03079222071,    3.29641209467,  -0.132094735941,    'relative_z',   -0.0064336403193],
                ['G', 0.190539062362,   0.465722471326, -0.848044676347,    'relative_x',   -0.642511925945],
                ['G', 0.721551907706,   1.16381288321,  -0.772996339157,    'relative_x',   -0.289776706887],
                ['G', 1.47101763085,    1.60984591744,  -0.394899370156,    'relative_x',   -0.267763428852],
                #['G', 3.00707952206,    3.2262991869,   -0.904680281084,    'relative_x',   -0.749314754922],
                ['G', 0.220360571969,   0.666490154733, -0.486977073953,    'relative_y',   -0.207559262548],
                ['G', 1.46298461486,    1.65338843085,  -0.203816256531,    'relative_y',   -0.0573698772288],
                ['G', 1.09474675239,    1.38525462141,  -0.314909447348,    'relative_y',   -0.057721530438],
                #['G', 3.00215145697,    3.5,            -0.555889236332,    'relative_y',   -0.356709192582],
                ['G', 0.209535706662,   0.737499193911, -0.124874036697,    'relative_z',   0.0683926377766],
                ['G', 1.23725140296,    1.66665509954,  -0.0735276194585,   'relative_z',   0.0718479736165], # End of taker
                ['F', 1.6708530028,     1.87993932525,  0.121374883191,     'taker_x',      0.242007104894], # Taker constraints
                ['F', 1.05773857322,    1.5079476125,   0.0970959826846,    'taker_x',      0.223143404052],
                ['F', 2.05026360364,    2.8750911438,   0.286725246807,     'taker_x',      0.396886561301],
                ['F', 0.686091983339,   1.19763915397,  -0.0259770061236,   'taker_y',      0.168932624124],
                ['F', 1.5406817201,     1.93212620428,  0.0166845226391,    'taker_y',      0.134555971478],
                #['F', 2.49839810756,    2.95847939429,  0.203480810414,     'taker_y',      0.327935547719],
                ['F', 0.963903148476,   1.38223034065,  -0.148030894276,    'taker_z',      -0.00591649758393],
                ['F', 2.02683257889,    2.71596926637,  -0.203546835998,    'taker_z',      -0.0928674678397],
                ['G', 0.227085583538,   0.907843927248, 0.103728187772,     'taker_x',      0.466003155728],
                ['G', 1.06191855894,    1.86016096711,  0.0828102661335,    'taker_x',      0.32349833853],
                ['G', 1.82697002397,    2.23611648473,  0.12487420768,      'taker_x',      0.33433001692],
                ['G', 0.159299721335,   0.933197095123, -0.0664540807286,   'taker_y',      0.217161596837],
                ['G', 1.26675860758,    1.69482074441,  -0.0478148264034,   'taker_y',      0.169387827968],
                ['G', 1.92309858155,    2.48624663659,  0.0424427366495,    'taker_y',      0.321822482734],
                ['G', 0.368050271527,   0.574281262842, -0.314900819025,    'taker_z',      -0.117475116225],
                ['G', 0.733054689208,   1.31906770443,  -0.307139280695,    'taker_z',      0.0164110473025],
                ['G', 1.43050963158,    1.77497291945,  -0.148560114954,    'taker_z',      -0.00650868033218],
                #['G', 2.7166769105,     2.90896962123,  -0.313985422969,    'taker_z',      -0.142684106931],
                ['G', 2.35733208497,    2.62499928801,  -0.281072361159,    'taker_z',      -0.0637040395312],
                ['F', 1.04166648675,    1.91883556964,  -0.0621640349839,   'taker_dx',     0.0507853098568],
                ['F', 1.65124540722,    2.15189988414,  0.080089526774,     'taker_dx',     0.197186961224],
                ['F', 2.03528546636,    2.87500158225,  0.173820377258,     'taker_dx',     0.287283880606],
                ['F', 0.691356772674,   1.47117917206,  -0.146553471892,    'taker_dy',     -0.0331193190845],
                ['F', 1.63074643665,    2.34437452005,  0.0585008719676,    'taker_dy',     0.176944534623],
                ['F', 1.13528201013,    1.63773160443,  -0.0536677382649,   'taker_dz',     0.0623555191736],
                ['F', 2.08333173494,    2.51391978727,  -0.223841369256,    'taker_dz',     -0.104389010036],
                ['G', 0.452937226225,   0.605701005177, -0.426481055693,    'taker_dx',     0.0376193134355],
                ['G', 0.798163283054,   1.03931666697,  -0.606399903335,    'taker_dx',     0.00844060801651],
                ['G', 1.72422371833,    2.1242441272,   -0.0166982135737,   'taker_dx',     0.336396880176],
                ['G', 2.27685477636,    2.99045267302,  -0.0687298069368,   'taker_dx',     0.433678688731],
                ['G', 0.250011995008,   0.681243487532, -0.358735419103,    'taker_dy',     0.0470185212601],
                ['G', 1.3221823993,     1.45828212676,  -0.205853013456,    'taker_dy',     0.203014994337],
                ['G', 1.60320366553,    1.95597004371,  -0.160512598808,    'taker_dy',     0.495934053596],
                ['G', 2.22205504274,    2.89630580521,  -0.0700187136945,   'taker_dy',     0.446051443314],
                ['G', 0.66866725203,    0.851532252702, -0.0279586928194,   'taker_dz',     0.499745533948],
                ['G', 1.75782958606,    2.07877539535,  -0.234257208251,    'taker_dz',     0.0704831446878],
                ['G', 2.210119463,      2.55382916601,  -0.304230617606,    'taker_dz',     0.0338071142025],
                ]

    #STL_list = [['F',0.1,1.0,-1,'giver_x',1]]

    def parse_STL_list(STL_list, dt):
        # Takes STL_list and outputs an STLFormula

        # Indexes used in STLGenerateSignal to identify variables in s[t][i]
        #Var_indexes = {'giver_x': 0, 'giver_y':1, 'giver_z':2, 'giver_dx': 3, 'giver_dy':4, 'giver_dz':5, 'relative_x':6, 'relative_y':7, 'relative_z':8, 'dxx':9, 'dyy':10, 'dzz':11, 'ax':12, 'ay':13, 'dax':14, 'day':15} # FLAGFLAGFLAG
        #Var_indexes = {'giver_x': 0, 'giver_y':1, 'giver_z':2, 'giver_dx': 3, 'giver_dy':4, 'giver_dz':5, 'taker_x':6, 'taker_y':7, 'taker_z':8, 'taker_dx':9, 'taker_dy':10, 'taker_dz':11, 'relative_x':12, 'relative_y':13, 'relative_z':14, 'K_p':15} # FLAGFLAGFLAG
        #Var_indexes = {'giver_x': 0, 'giver_y':1, 'giver_z':2, 'giver_dx': 0, 'giver_dy':1, 'giver_dz':2, 'relative_x':0, 'relative_y':1, 'relative_z':2}

        formula = []

        # Create each predicate in phi
        for i in range(len(STL_list)):
            variable_name = STL_list[i][4]
            variable_index = dimensions.index(variable_name)
            Spatial_predicate = STLFormula.Conjunction(
                STLFormula.Predicate(variable_name, operatorclass.gt, STL_list[i][3], variable_index),
                STLFormula.Predicate(variable_name, operatorclass.lt, STL_list[i][5], variable_index)
            )
            t1 = int(STL_list[i][1] / dt) # Round down
            t2 = int(STL_list[i][2] / dt) + 1 # Round up
            if STL_list[i][0] == 'F':
                formula.append(STLFormula.Eventually(Spatial_predicate, t1, t2))
            elif STL_list[i][0] == 'G':
                formula.append(STLFormula.Always(Spatial_predicate, t1, t2))
            else:
                raise 'Bad STL_list'

        # Join phi in a conjunction
        def conjoin_list(formula):
            if len(formula) == 1:
                return formula[0]
            else:
                return STLFormula.Conjunction(formula[0], conjoin_list(formula[1:]))

        return conjoin_list(formula)

    phi = parse_STL_list(STL_list, dt)

    # Kshirsagar approach, towards human
    #   position only, no orientation
    #   l1 norm
    # F[0, t] ( ||p_h - p_r - p_d|| < epsilon )
    KSHIRSAGAR_DELTA = [0.3, 0.1, 0.0]# x,y,z offset of giver and taker hands on the object
    KSHIRSAGAR_EPSILON = 0.01
    KSHIRSAGAR_TIME = 2.5 # [s]

    t_K = int(KSHIRSAGAR_TIME / dt) + 1
    Spatial_predicate = STLFormula.Predicate('K_p', operatorclass.lt, KSHIRSAGAR_EPSILON, 9) # Fix variable pi_index_signal = 9 if number of signals change
    Kshirsagar_part = STLFormula.Eventually( Spatial_predicate, 0, t_K)
    if KSHIRSAGAR_APPROACH == True:
        phi = STLFormula.Conjunction(phi, Kshirsagar_part)

    #parameters
    start=[human_data.x_giver_RHand[0], human_data.y_giver_RHand[0], human_data.z_giver_RHand[0], # Giver position
           human_data.x_taker_RHand[0], human_data.y_taker_RHand[0], human_data.z_taker_RHand[0], # Taker position
           ]
    domain=[-2, 2] # Domain of signals
    max_speed = 1 # m/s
    # Max acceleration
    #          Max    Mean   TrmMen 90%:ile
    # Giver  ̈x 6.8506 3.4281 2.5807 6.2893
    # Giver  ̈y 3.8406 2.0545 1.8066 3.2141
    # Giver  ̈z 8.3751 4.5666 3.8804 7.6749
    # Taker  ̈x 6.1479 1.9960 1.8423 2.5355
    # Taker  ̈y 4.2769 1.8407 1.5427 3.2640
    # Taker  ̈z 7.7459 2.3012 1.5694 4.7213
    max_giver_acceleration = [6.8506, 3.8406, 8.3751] # m/s^2
    max_taker_acceleration = [6.1479, 4.2769, 7.7459] # m/s^2
    max_human_speed = 1 # m/s
    max_change = [0,0,0, 
                  max_giver_acceleration[0] * dt, max_giver_acceleration[1] * dt, max_giver_acceleration[2] * dt, 
                  0,0,0,
                  max_taker_acceleration[0] * dt, max_taker_acceleration[1] * dt, max_taker_acceleration[2] * dt]
    
    phi_nnf = STLFormula.toNegationNormalForm(phi, False)
    

    # Initiate optimization problem for generating signals
    problem = generate_signal_problem(phi, start, domain, dimensions, max_change, dt, 
                                      OPTIMIZE_ROBUSTNESS=OPTIMIZE_ROBUSTNESS,
                                      QUANTITATIVE_OPTIMIZATION= QUANTITATIVE_OPTIMIZATION,
                                      HARD_STL_CONSTRAINT= HARD_STL_CONSTRAINT, 
                                      KSHIRSAGAR_APPROACH= KSHIRSAGAR_APPROACH,
                                      KSHIRSAGAR_DELTA = KSHIRSAGAR_DELTA )

    # Structure feedback / human motion prediction data
    # Resample human data to dt sample time
    idx = [min(enumerate(human_data.time), key=lambda x: abs(target - x[1]))[0] for target in [x*dt for x in range(problem.phi.horizon * 2)]] # find closest time index!
    feedback = np.array([human_data.x_taker_RHand[idx], human_data.y_taker_RHand[idx], human_data.z_taker_RHand[idx]]).T
    true_giver = np.array([human_data.x_giver_RHand[idx], human_data.y_giver_RHand[idx], human_data.z_giver_RHand[idx]]).T
    true_taker = feedback

    # First execution
    start_execution_time = time.monotonic()
    trajectory1 = problem.generate_path()
    execution_time = time.monotonic() - start_execution_time
    current_step = 0

    # Get mean square error
    [giver_mse, taker_mse] = problem.prediction_mse(true_giver, true_taker, t_start = current_step)

    # Plot trajectory
    plot_trajectory(phi, current_step, trajectory1, human_data, dt, giver_mse, taker_mse, save_flag = True )

    # Online (Real time) sim vs Offline simulation (recompute every step)
    ONLINE_SIM = False 
    current_step = 0
    sim_time = 0.0
    Handover_done = False
    human_position_idx = [6,7,8]


    simulation_timer_start = time.monotonic()

    # Execution loop
    while not Handover_done:
        previous_step = current_step
        # get current time and step
        if ONLINE_SIM:
            current_sim_time = time.monotonic() - simulation_timer_start
            current_step = floor( (current_sim_time) / dt )
        else:
            current_step += 1
            current_sim_time = current_step * dt
        print("Current time = " + str(current_sim_time))

        

        # Fix past variables (robot movement, human position)
        # Assume robot moved exactly as planned
        problem.fix_variables(range(previous_step+1, current_step+1), range(0,3)) 
        # Set human motion from recorded handover
        problem.fix_variables(range(previous_step+1, current_step+1), human_position_idx, feedback[range(previous_step+1, current_step+1)].tolist() )

        # (later) Fix predicted human motion window

        # Generate new path
        trajectory1 = problem.generate_path()

        # (later) Unfix predicted human motion window

        
        # Get mean square error
        [giver_mse, taker_mse] = problem.prediction_mse(true_giver, true_taker, t_start = current_step)

        # Update plot?
        plot_trajectory(phi, current_step, trajectory1, human_data, dt, giver_mse, taker_mse)

        # Break loop if exceeding time
        if current_step >= problem.phi.horizon:
            Handover_done = True
            print("Handover done!!!")
            break

        ## Break loop if done (Detect loadshare???) FLAGFLAGFLAG
        #if False:
        #    Handover_done = True
    
    #problem.fix_variables([0,1], [1,2], [[1.337, 0.69], [0.420, 0.80085]])
    #problem.unfix_variables([0,1], [1])

    # Make patches to draw in graphs
    x_patches = []
    y_patches = []
    z_patches = []
    dx_patches = []
    dy_patches = []
    dz_patches = []
    xx_patches = []
    yy_patches = []
    zz_patches = []
    codes = [Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
            ]
    for i in range(len(STL_list)):
        # Psi_[t1, t2] (lb < var < ub)
        verts = [
            (STL_list[i][1], STL_list[i][3]), # left, bottom
            (STL_list[i][1], STL_list[i][5]), # left, top
            (STL_list[i][2], STL_list[i][5]), # right, top
            (STL_list[i][2], STL_list[i][3]), # right, bottom
            (0., 0.), # ignored
            ]
        path = Path(verts, codes)

        #patch = plt.Rectangle((STL_list[i][1], STL_list[i][2]), STL_list[i][3], STL_list[i][5])
        
        if STL_list[i][0] == 'F':
            pp = patches.PathPatch(path, facecolor='lightsalmon',lw=0, alpha=0.5)
        elif STL_list[i][0] == 'G':
            pp = patches.PathPatch(path, facecolor='palegreen',lw=0, alpha=0.5)

        if STL_list[i][4] == 'giver_x':
            x_patches.append(pp)
        elif STL_list[i][4] == 'giver_y':
            y_patches.append(pp)
        elif STL_list[i][4] == 'giver_z':
            z_patches.append(pp)
        elif STL_list[i][4] == 'giver_dx':
            dx_patches.append(pp)
        elif STL_list[i][4] == 'giver_dy':
            dy_patches.append(pp)
        elif STL_list[i][4] == 'giver_dz':
            dz_patches.append(pp)
        elif STL_list[i][4] == 'relative_x':
            xx_patches.append(pp)
        elif STL_list[i][4] == 'relative_y':
            yy_patches.append(pp)
        elif STL_list[i][4] == 'relative_z':
            zz_patches.append(pp)
        # FLAGFLAGFLAG add more patch variables to this (maybe change to a switch/case)

    

    t_trajectory = np.linspace(0, phi.horizon*dt, phi.horizon+1)
    t_human = list(human_data.time[::DOWNSAMPLING])

    if PLOT_POSITION:
        fig_pos = plt.figure()
        ax_pos_x = fig_pos.add_subplot(131)
        plt.grid(True)
        ax_pos_y = fig_pos.add_subplot(132)
        plt.grid(True)
        ax_pos_z = fig_pos.add_subplot(133)
        plt.grid(True)
        
        # x
        if trajectory1:
            x_giver = [x[0] for x in trajectory1]
            x_taker = [x[6] for x in trajectory1]
            ax_pos_x.plot(t_trajectory, x_giver, '-g', marker='o', label=r'$\rho='+str(round(phi.robustness(trajectory1,0),3))+'$')
            ax_pos_x.plot(t_trajectory, x_taker, '-g', marker='o')
        if PLOT_HUMAN_GIVER:
            ax_pos_x.plot(t_human, human_data.x_giver_RHand[::DOWNSAMPLING], '-m', alpha=0.5, label=r'human giver')
        if PLOT_HUMAN_TAKER:
            ax_pos_x.plot(t_human, human_data.x_taker_RHand[::DOWNSAMPLING], '-b', alpha=0.5, label=r'human taker')
        ax_pos_x.set_xlabel('t')
        ax_pos_x.set_ylabel('X')
        for patch in x_patches:
            ax_pos_x.add_patch(patch)

        # y
        if trajectory1:
            x_giver = [x[1] for x in trajectory1]
            x_taker = [x[7] for x in trajectory1]
            ax_pos_y.plot(t_trajectory, x_giver, '-g', marker='o', label=r'$\rho='+str(round(phi.robustness(trajectory1,0),3))+'$; $t='+str(round(execution_time,3))+'$')
            ax_pos_y.plot(t_trajectory, x_taker, '-g', marker='o')
        if PLOT_HUMAN_GIVER:
            ax_pos_y.plot(t_human, human_data.y_giver_RHand[::DOWNSAMPLING], '-m', alpha=0.5, label=r'human giver')
        if PLOT_HUMAN_TAKER:
            ax_pos_y.plot(t_human, human_data.y_taker_RHand[::DOWNSAMPLING], '-b', alpha=0.5, label=r'human taker')
        ax_pos_y.set_xlabel('t')
        ax_pos_y.set_ylabel('Y')
        for patch in y_patches:
            ax_pos_y.add_patch(patch)
    
        # z
        if trajectory1:
            x_giver = [x[2] for x in trajectory1]
            x_taker = [x[8] for x in trajectory1]
            ax_pos_z.plot(t_trajectory, x_giver, '-g', marker='o', label=r'$\rho='+str(round(phi.robustness(trajectory1,0),3))+'$')
            ax_pos_z.plot(t_trajectory, x_taker, '-g', marker='o')
        if PLOT_HUMAN_GIVER:
            ax_pos_z.plot(t_human, human_data.z_giver_RHand[::DOWNSAMPLING], '-m', alpha=0.5, label=r'human giver')
        if PLOT_HUMAN_TAKER:
            ax_pos_z.plot(t_human, human_data.z_taker_RHand[::DOWNSAMPLING], '-b', alpha=0.5, label=r'human taker')
        ax_pos_z.set_xlabel('t')
        ax_pos_z.set_ylabel('Z')
        for patch in z_patches:
            ax_pos_z.add_patch(patch)
    
        ax_pos_y.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=3)
        #fig_pos.tight_layout()

          # MSE
        [giver_mse, taker_mse] = problem.prediction_mse(true_giver, true_taker)
        fig_pos.text(0.98, 0.98, # 'Giver mse:\n $x={0:.3f}$\n $y={1:.3f}$\n $z={2:.3f}$\n $xyz={3:.3f}$'.format(1,2,3,4))
                "Giver mse: $x = {:.4f}, y = {:.4f}, z = {:.4f}, [x,y,z]^T = {:.4f}$\n".format(giver_mse[0],giver_mse[1],giver_mse[2],giver_mse[3])
                + 'Taker mse: $x = {:.4f}, y = {:.4f}, z = {:.4f}, [x,y,z]^T = {:.4f}$'.format(taker_mse[0],taker_mse[1],taker_mse[2],taker_mse[3]),
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(boxstyle="round",facecolor='white',edgecolor='black')
        )
        plt.close()

        plot_trajectory(phi, problem.phi.horizon, trajectory1, human_data, dt, giver_mse, taker_mse, save_flag = False, close_flag = False )

        plt.savefig( PLOT_PATH + PLOT_NAME + 'Full' )
        plt.show()
