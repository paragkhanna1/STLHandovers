#!/usr/bin/env python

from STLGenerateJointHandover import generate_signal_problem, plot_trajectory, plot_trajectory2, plot_in_worldframe
import pandas as pd
from math import floor, sqrt
from STL import STLFormula
import operator as operatorclass
#import pulp as plp
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
import time
from os.path import lexists
#from sklearn.metrics import mean_squared_error as mse
from tfmatrix import points2tfmatrix, points2invtfmatrix
import os
import pulp as plp

# Commented for test without ros
import rospy
import math
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf
import geometry_msgs.msg
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt
import tf.transformations as tr

#MarkerArray marker
from copy import deepcopy
import csv


import pandas as pd # pandas==2.0.3
broadcaster_robot_desired_pose = tf2_ros.StaticTransformBroadcaster() 

UPDATE_ROBOT_FROM_PLAN = True
UPDATE_HUMAN_FROM_PLAN = False
 
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)

class plotRecordedHandover:
    def __init__(self, dt, robot_role):
        self.dt = dt
        self.robot_role = robot_role

    def set_variables_from_lists(self, x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, 
                                       x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, 
                                       x_giver_chest_list, y_giver_chest_list, z_giver_chest_list,
                                       x_taker_chest_list, y_taker_chest_list, z_taker_chest_list,
                                       predicted_handover_n_list, predicted_handover_K_list,
                                       trajectory_list,
                                       dimensions,
                                       x_true_giver_list,
                                       y_true_giver_list,
                                       z_true_giver_list,
                                       x_true_taker_list,
                                       y_true_taker_list,
                                       z_true_taker_list):
        """
            Every variable will be saved as np.array, self.variable_name[i][n]
                where i selects which plan, n is the step 

        """
        self.dimensions = dimensions

        self.x_giver_in_mid = np.array(x_giver_in_mid)
        self.y_giver_in_mid = np.array(y_giver_in_mid)
        self.z_giver_in_mid = np.array(z_giver_in_mid)
        self.x_taker_in_mid = np.array(x_taker_in_mid)
        self.y_taker_in_mid = np.array(y_taker_in_mid)
        self.z_taker_in_mid = np.array(z_taker_in_mid)

        self.x_giver_chest_list = np.array(x_giver_chest_list)
        self.y_giver_chest_list = np.array(y_giver_chest_list)
        self.z_giver_chest_list = np.array(z_giver_chest_list)
        self.x_taker_chest_list = np.array(x_taker_chest_list)
        self.y_taker_chest_list = np.array(y_taker_chest_list)
        self.z_taker_chest_list = np.array(z_taker_chest_list)
        
        self.x_true_giver_list = x_true_giver_list
        self.y_true_giver_list = y_true_giver_list
        self.z_true_giver_list = z_true_giver_list
        self.x_true_taker_list = x_true_taker_list
        self.y_true_taker_list = y_true_taker_list
        self.z_true_taker_list = z_true_taker_list

        if self.robot_role == 'Taker':
            robot_name = 'taker'
            human_name = 'giver'
        else:
            robot_name = 'giver'
            human_name = 'taker'



        # trajectory_list[t_recalculate][t][dim]
        trajectory_list = np.array( trajectory_list )
        # velocity
        self.dx_giver_in_mid = trajectory_list[:,:, dimensions.index(human_name + '_dx') ]
        self.dy_giver_in_mid = trajectory_list[:,:, dimensions.index(human_name + '_dy') ]
        self.dz_giver_in_mid = trajectory_list[:,:, dimensions.index(human_name + '_dz') ]
        self.dx_taker_in_mid = trajectory_list[:,:, dimensions.index(robot_name + '_dx') ]
        self.dy_taker_in_mid = trajectory_list[:,:, dimensions.index(robot_name + '_dy') ]
        self.dz_taker_in_mid = trajectory_list[:,:, dimensions.index(robot_name + '_dz') ]

        ## acceleration
        self.ddx_giver_in_mid = trajectory_list[:,:, dimensions.index(human_name + '_ddx') ]
        self.ddy_giver_in_mid = trajectory_list[:,:, dimensions.index(human_name + '_ddy') ]
        self.ddz_giver_in_mid = trajectory_list[:,:, dimensions.index(human_name + '_ddz') ]
        self.ddx_taker_in_mid = trajectory_list[:,:, dimensions.index(robot_name + '_ddx') ]
        self.ddy_taker_in_mid = trajectory_list[:,:, dimensions.index(robot_name + '_ddy') ]
        self.ddz_taker_in_mid = trajectory_list[:,:, dimensions.index(robot_name + '_ddz') ]

        ## K
        self.K = trajectory_list[:,:, dimensions.index('K_p') ]


        self.N = self.x_giver_in_mid.shape[1] # Length of one trajectory
        self.M = self.x_giver_in_mid.shape[0] # Number of recorded trajectories
        self.no_trajectories = len(x_giver_chest_list) # Number of recorded trajectories

        self.predicted_handover_n_list = predicted_handover_n_list
        self.predicted_handover_K_list = predicted_handover_K_list

    def set_variables_from_csv(self, filepath): # UNRELIABLE
        """
            Every variable will be saved as np.array, self.variable_name[i][n]
                where i selects which plan, n is the step 
            x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, human_chest_list, robot_chest_list
        """
        x_giver_in_mid = pd.read_csv(filepath + 'x_giver_in_mid.csv', sep=",\s+", engine='python', comment = ']', header = None)
        y_giver_in_mid = pd.read_csv(filepath + 'y_giver_in_mid.csv', sep=",\s+", engine='python', comment = ']', header = None)
        z_giver_in_mid = pd.read_csv(filepath + 'z_giver_in_mid.csv', sep=",\s+", engine='python', comment = ']', header = None)
        x_taker_in_mid = pd.read_csv(filepath + 'x_rob_in_mid.csv', sep=",\s+", engine='python', comment = ']', header = None)
        y_taker_in_mid = pd.read_csv(filepath + 'y_rob_in_mid.csv', sep=",\s+", engine='python', comment = ']', header = None)
        z_taker_in_mid = pd.read_csv(filepath + 'z_rob_in_mid.csv', sep=",\s+", engine='python', comment = ']', header = None)
    
        # delim_whitespace = True, , sep="[\s+]", engine='python', comment = ']'
        # , engine='c', lineterminator='[', comment=']', decimal='.'
        human_chest_list = pd.read_csv(filepath + 'human_chest_list.csv', delim_whitespace=True, engine='python', comment = ']', header = None)
        robot_chest_list = pd.read_csv(filepath + 'robot_chest_list.csv', delim_whitespace=True, engine='python', comment = ']', header = None)

        self.N = x_giver_in_mid.shape[1] # Length of one trajectory
        self.M = x_giver_in_mid.shape[0] # Number at end of recorded data ###NEW
        self.no_trajectories = len(human_chest_list) # Number of recorded trajectories

        self.x_giver_in_mid = x_giver_in_mid.values
        self.y_giver_in_mid = y_giver_in_mid.values
        self.z_giver_in_mid = z_giver_in_mid.values
        self.x_taker_in_mid = x_taker_in_mid.values
        self.y_taker_in_mid = y_taker_in_mid.values
        self.z_taker_in_mid = z_taker_in_mid.values
        self.x_giver_in_mid[:,0] = [ float(self.x_giver_in_mid[i][0][1:]) for i in range(0,self.no_trajectories) ] # First value reads bad
        self.y_giver_in_mid[:,0] = [ float(self.y_giver_in_mid[i][0][1:]) for i in range(0,self.no_trajectories) ] # Changes first value from string to float
        self.z_giver_in_mid[:,0] = [ float(self.z_giver_in_mid[i][0][1:]) for i in range(0,self.no_trajectories) ]
        self.x_taker_in_mid[:,0] = [ float(self.x_taker_in_mid[i][0][1:]) for i in range(0,self.no_trajectories) ]
        self.y_taker_in_mid[:,0] = [ float(self.y_taker_in_mid[i][0][1:]) for i in range(0,self.no_trajectories) ]
        self.z_taker_in_mid[:,0] = [ float(self.z_taker_in_mid[i][0][1:]) for i in range(0,self.no_trajectories) ]

        self.x_giver_chest_list = np.array([ float(human_chest_list.values[i][0][1:]) for i in range(0,self.no_trajectories)])
        self.y_giver_chest_list = np.array([ human_chest_list.values[i][1] for i in range(0,self.no_trajectories)])
        self.z_giver_chest_list = np.array([ human_chest_list.values[i][2] for i in range(0,self.no_trajectories)])

        self.x_taker_chest_list = np.array([ float(robot_chest_list.values[i][0][1:]) for i in range(0,self.no_trajectories)])
        self.y_taker_chest_list = np.array([ robot_chest_list.values[i][1] for i in range(0,self.no_trajectories)])
        self.z_taker_chest_list = np.array([ robot_chest_list.values[i][2] for i in range(0,self.no_trajectories)])

        return True
    
    def set_robot_role(self, robot_role):
        if robot_role in ['Giver', 'Taker']:
            self.robot_role = robot_role
        else:
            print("valid roles for robot are ['Giver', 'Taker']")

        
    def plot_all_handovers(self, show_plots = True, save_plots = False, save_location = ''):
        for trajectory_idx in range(0,self.no_trajectories):
            
            if not(os.path.exists(save_location + 'world_frame')):
                os.mkdir(save_location + 'world_frame')
            if not(os.path.exists(save_location + 'shared_frame')):
                os.mkdir(save_location + 'shared_frame')

            # Plot with labels "Giver" and "Taker"
            #self.plot_one_handover(trajectory_idx, world_frame = True)
            #if save_plots:
            #    #plt.show(block=True)
            #    plt.savefig( save_location + 'world_frame/' + 'plan_world_frame_pos' + str(trajectory_idx) + '.pdf' )
            #    plt.close()
                
            # Plot with labels "Human" and "Robot"
            self.plot_one_handover2(trajectory_idx, world_frame = True)
            if save_plots:
                #plt.show(block=True)
                plt.savefig( save_location + 'world_frame/' + 'plan_world_frame_pos' + str(trajectory_idx) + '.pdf' )
                plt.close()

            #self.plot_one_handover_velocity(trajectory_idx, world_frame = True)
            #if save_plots:
            #    plt.savefig( save_location + 'world_frame/' + 'plan_world_frame_vel' + str(trajectory_idx) + '.pdf' )
            #    plt.close()
            #self.plot_one_handover_acceleration(trajectory_idx, world_frame = True)
            #if save_plots:
            #    plt.savefig( save_location + 'world_frame/' + 'plan_world_frame_acc' + str(trajectory_idx) + '.pdf' )
            #    plt.close()
            #self.plot_one_handover(trajectory_idx, world_frame = False)
            #if save_plots:
            #    plt.savefig( save_location + 'shared_frame/' + 'plan_share_frame_pos' + str(trajectory_idx) + '.pdf' )
            #    plt.close()
            #self.plot_one_handover_velocity(trajectory_idx, world_frame = False)
            #if save_plots:
            #    plt.savefig( save_location + 'shared_frame/' + 'plan_share_frame_vel' + str(trajectory_idx) + '.pdf' )
            #    plt.close()
            #self.plot_one_handover_acceleration(trajectory_idx, world_frame = False)
            #if save_plots:
            #    plt.savefig( save_location + 'shared_frame/' + 'plan_share_frame_acc' + str(trajectory_idx) + '.pdf' )
            #    plt.close()
        if show_plots:
            plt.show()

    
    def plot_one_handover_velocity(self, trajectory_idx, current_step=0, world_frame = True):
        """
            Plot in worldframe, a trajectory given in shared frame
        """
        current_step = trajectory_idx # TODO
        # place recording better in the statemachine to get first plan

        #if self.robot_role == 'Giver':
        #    chest_Giver = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        #    chest_Taker = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )
        #elif self.robot_role == 'Taker':
        #    chest_Taker = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        #    chest_Giver = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )
        chest_Taker = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        chest_Giver = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )

        xg = self.dx_giver_in_mid[trajectory_idx, :]
        yg = self.dy_giver_in_mid[trajectory_idx, :]
        zg = self.dz_giver_in_mid[trajectory_idx, :]
        xt = self.dx_taker_in_mid[trajectory_idx, :]
        yt = self.dy_taker_in_mid[trajectory_idx, :]
        zt = self.dz_taker_in_mid[trajectory_idx, :]
        xg_world = np.zeros(xg.shape)
        yg_world = np.zeros(yg.shape)
        zg_world = np.zeros(zg.shape)
        xt_world = np.zeros(xt.shape)
        yt_world = np.zeros(yt.shape)
        zt_world = np.zeros(zt.shape)

        #final_xg = self.dx_giver_in_mid[-1, :]
        #final_yg = self.dy_giver_in_mid[-1, :]
        #final_zg = self.dz_giver_in_mid[-1, :]
        #final_xt = self.dx_taker_in_mid[-1, :]
        #final_yt = self.dy_taker_in_mid[-1, :]
        #final_zt = self.dz_taker_in_mid[-1, :]
        #final_xg_world = np.zeros(xg.shape)
        #final_yg_world = np.zeros(yg.shape)
        #final_zg_world = np.zeros(zg.shape)
        #final_xt_world = np.zeros(xt.shape)
        #final_yt_world = np.zeros(yt.shape)
        #final_zt_world = np.zeros(zt.shape)

        handover_point = self.predicted_handover_n_list[trajectory_idx]
        handover_distance = self.predicted_handover_K_list[trajectory_idx]

        if world_frame:
            # Transform
            for t in range(self.N):
                if t < self.no_trajectories:
                    TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
                    ROT = TF[0:3, 0:3]
                P_n = np.matmul(ROT, np.array([xg[t], yg[t], zg[t]]))  #giver point at time n
                xg_world[t] = P_n[0]
                yg_world[t] = P_n[1]
                zg_world[t] = P_n[2] # TF to world frame

                P_n = np.matmul(ROT, np.array([xt[t], yt[t], zt[t]]))  #taker point at time n
                xt_world[t] = P_n[0]
                yt_world[t] = P_n[1]
                zt_world[t] = P_n[2] # TF to world frame
            
                #P_n = np.matmul(ROT, np.array([final_xg[t], final_yg[t], final_zg[t]]))  #final trajectory of handover, human
                #final_xg_world[t] = P_n[0]
                #final_yg_world[t] = P_n[1]
                #final_zg_world[t] = P_n[2] # TF to world frame

                #P_n = np.matmul(ROT, np.array([final_xt[t], final_yt[t], final_zt[t]]))  #final trajectory of handover, robot
                #final_xt_world[t] = P_n[0]
                #final_yt_world[t] = P_n[1]
                #final_zt_world[t] = P_n[2] # TF to world frame
        else:
            # Don't transform
            xg_world = xg
            yg_world = yg
            zg_world = zg
            xt_world = xt
            yt_world = yt
            zt_world = zt
            #final_xg_world = final_xg
            #final_yg_world = final_yg
            #final_zg_world = final_zg
            #final_xt_world = final_xt
            #final_yt_world = final_yt
            #final_zt_world = final_zt


        # # # # plot the figure # # # #
        # make figure
        fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['legend',   'legend',   'legend'],
                                               ['legend',   'legend',   'legend']], layout='tight')
        ax_pos_x = ax_dict['x']
        ax_pos_y = ax_dict['y']
        ax_pos_zh = ax_dict['zh_plot']
        ax_pos_zr = ax_dict['zr_plot']
        ax_pos_x.grid(True)
        ax_pos_y.grid(True)
        ax_pos_zh.grid(True)
        ax_pos_zr.grid(True)

        t = np.linspace(0, (self.N-1)*self.dt, self.N)
        
        # plot x
        ax_pos_x.plot(t[0:current_step+1], xg_world[0:current_step+1], '-g', marker='o', label='Giver velocity')
        ax_pos_x.plot(t[current_step:handover_point+1], xg_world[current_step:handover_point+1], '--g', marker='x', label='Giver velocity')        
        #ax_pos_x.plot(t[handover_point:self.N], xg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Giver velocity')        
    
        ax_pos_x.plot(t[0:current_step+1], xt_world[0:current_step+1], '-r', marker='o', label='Taker velocity')
        ax_pos_x.plot(t[current_step:handover_point+1], xt_world[current_step:handover_point+1], '--r', marker='x', label='Taker planed velocity')
        #ax_pos_x.plot(t[handover_point:self.N], xt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Taker planed velocity')

        # Best handover location
        if handover_distance <= 0.1:
            ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        else:
            ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )

        #ax_pos_x.plot(t, final_xg_world, '--m', alpha=0.5, label='Final human velocity')
        #ax_pos_x.plot(t, final_xt_world, '--b', alpha=0.5, label='Final robot velocity')

        ax_pos_x.set_xlabel('t [s]')
        ax_pos_x.set_ylabel('dX [m/s]')
    
        # plot y
        ax_pos_y.plot(t[0:current_step+1], yg_world[0:current_step+1], '-g', marker='o', label='Giver velocity')
        ax_pos_y.plot(t[current_step:handover_point+1], yg_world[current_step:handover_point+1], '--g', marker='x', label='Giver plan')
        #ax_pos_y.plot(t[handover_point:self.N], yg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Giver plan (past handover)')
        ax_pos_y.plot(t[0:current_step+1], yt_world[0:current_step+1], '-r', marker='o', label='Taker velocity')
        ax_pos_y.plot(t[current_step:handover_point+1], yt_world[current_step:handover_point+1], '--r', marker='x', label='Taker plan')
        #ax_pos_y.plot(t[handover_point:self.N], yt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Taker plan (past handover)')
      
        #ax_pos_y.plot(t, final_yg_world, '--m', alpha=0.5, label='Final human')
        #ax_pos_y.plot(t, final_yt_world, '--b', alpha=0.5, label='Final plan robot')
    
        # Best handover location
        #ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K = '+str( handover_distance ) )
        if handover_distance <= 0.1:
            ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + " = {:.2f}".format( handover_distance ) )
        else:
            ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + " = {:.2f}".format( handover_distance ) )
        
        ax_pos_y.set_xlabel('t [s]')
        ax_pos_y.set_ylabel('dY [m/s]')

        # plot z
        ax_pos_zh.plot(t[0:current_step+1], zg_world[0:current_step+1], '-g', marker='o', label='Giver')
        ax_pos_zh.plot(t[current_step:handover_point+1], zg_world[current_step:handover_point+1], '--g', marker='x', label='Giver')        
        #ax_pos_zh.plot(t[handover_point:self.N], zg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Giver')        
    
        #ax_pos_zh.plot(t, final_zg_world, '--m', alpha=0.5, label='Final human path')    

        ax_pos_zr.plot(t[0:current_step+1], zt_world[0:current_step+1], '-r', marker='o', label='Taker')
        ax_pos_zr.plot(t[current_step:handover_point+1], zt_world[current_step:handover_point+1], '--r', marker='x', label='Taker')
        #ax_pos_zr.plot(t[handover_point:self.N], zt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Taker')
    
        #ax_pos_zr.plot(t, final_zt_world, '--b', alpha=0.5, label='Final robot path')
    
        ax_pos_zh.set_xlabel('t [s]')
        ax_pos_zh.set_ylabel('dZ [m/s]')
        ax_pos_zr.set_xlabel('t [s]')
        ax_pos_zr.set_ylabel('dZ [m/s]')

        # Best handover location
        #ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K = '+str( handover_distance ) )
        #ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K = '+str( handover_distance ) )
        if handover_distance <= 0.1:
            #ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        else:
            #ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|r', linestyle='', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|r', linestyle='', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        

        # Title
        if world_frame:
            ax_pos_y.set_title('Handover velocity in world frame \n n = ' + str(current_step))
        else:
            ax_pos_y.set_title('Handover velocity in shared frame \n n = ' + str(current_step))

        # Legend
        ax_dict['legend'].set_visible(False)
        ax_dict['legend'].set_box_aspect(0.001)
        handles, labels = ax_pos_y.get_legend_handles_labels()
        fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=4)

        # Axis limits
        #ax_pos_x.set_ylim(0.0, 1.75)
        #ax_pos_y.set_ylim(-0.7, 0.7)
        #ax_pos_zh.set_ylim(-0.3, 0.4)
        #ax_pos_zr.set_ylim(-0.3, 0.4)

        ax_pos_x.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_y.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_zh.set_xlim(-0.2, (self.N+1) * self.dt) # Window size (important for consistent figures)
        ax_pos_zr.set_xlim(-0.2, (self.N+1) * self.dt) 
        
        fig_pos.set_figheight(6)
        fig_pos.set_figwidth(12)

    def plot_one_handover_acceleration(self, trajectory_idx, current_step=0, world_frame = True):
        """
            Plot in worldframe or share frame, a trajectory given in shared frame
        """
        current_step = trajectory_idx # TODO
        # place recording better in the statemachine to get first plan

        #if self.robot_role == 'Giver':
        #    chest_Giver = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        #    chest_Taker = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )
        #elif self.robot_role == 'Taker':
        #    chest_Taker = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        #    chest_Giver = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )
        chest_Giver = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        chest_Taker = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )

        xg = self.ddx_giver_in_mid[trajectory_idx, :]
        yg = self.ddy_giver_in_mid[trajectory_idx, :]
        zg = self.ddz_giver_in_mid[trajectory_idx, :]
        xt = self.ddx_taker_in_mid[trajectory_idx, :]
        yt = self.ddy_taker_in_mid[trajectory_idx, :]
        zt = self.ddz_taker_in_mid[trajectory_idx, :]
        xg_world = np.zeros(xg.shape)
        yg_world = np.zeros(yg.shape)
        zg_world = np.zeros(zg.shape)
        xt_world = np.zeros(xt.shape)
        yt_world = np.zeros(yt.shape)
        zt_world = np.zeros(zt.shape)

        #final_xg = self.ddx_giver_in_mid[-1, :]
        #final_yg = self.ddy_giver_in_mid[-1, :]
        #final_zg = self.ddz_giver_in_mid[-1, :]
        #final_xt = self.ddx_taker_in_mid[-1, :]
        #final_yt = self.ddy_taker_in_mid[-1, :]
        #final_zt = self.ddz_taker_in_mid[-1, :]
        #final_xg_world = np.zeros(xg.shape)
        #final_yg_world = np.zeros(yg.shape)
        #final_zg_world = np.zeros(zg.shape)
        #final_xt_world = np.zeros(xt.shape)
        #final_yt_world = np.zeros(yt.shape)
        #final_zt_world = np.zeros(zt.shape)

        handover_point = self.predicted_handover_n_list[trajectory_idx]
        handover_distance = self.predicted_handover_K_list[trajectory_idx]

        if world_frame:
            # Transform
            for t in range(self.N):
                if t < self.no_trajectories:
                    TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
                    ROT = TF[0:3, 0:3]
                P_n = np.matmul(ROT, np.array([xg[t], yg[t], zg[t]]))  #giver point at time n
                xg_world[t] = P_n[0]
                yg_world[t] = P_n[1]
                zg_world[t] = P_n[2] # TF to shared frame

                P_n = np.matmul(ROT, np.array([xt[t], yt[t], zt[t]]))  #taker point at time n
                xt_world[t] = P_n[0]
                yt_world[t] = P_n[1]
                zt_world[t] = P_n[2] # TF to shared frame

                #P_n = np.matmul(ROT, np.array([final_xg[t], final_yg[t], final_zg[t]]))  #final trajectory of handover, human
                #final_xg_world[t] = P_n[0]
                #final_yg_world[t] = P_n[1]
                #final_zg_world[t] = P_n[2] # TF to shared frame

                #P_n = np.matmul(ROT, np.array([final_xt[t], final_yt[t], final_zt[t]]))  #final trajectory of handover, robot
                #final_xt_world[t] = P_n[0]
                #final_yt_world[t] = P_n[1]
                #final_zt_world[t] = P_n[2] # TF to shared frame
        else:
            # Don't transform
            xg_world = xg
            yg_world = yg
            zg_world = zg
            xt_world = xt
            yt_world = yt
            zt_world = zt
            #final_xg_world = final_xg
            #final_yg_world = final_yg
            #final_zg_world = final_zg
            #final_xt_world = final_xt
            #final_yt_world = final_yt
            #final_zt_world = final_zt
                
        # # # # plot the figure # # # #
        # make figure
        fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['legend',   'legend',   'legend'],
                                               ['legend',   'legend',   'legend']], layout='tight')
        ax_pos_x = ax_dict['x']
        ax_pos_y = ax_dict['y']
        ax_pos_zh = ax_dict['zh_plot']
        ax_pos_zr = ax_dict['zr_plot']
        ax_pos_x.grid(True)
        ax_pos_y.grid(True)
        ax_pos_zh.grid(True)
        ax_pos_zr.grid(True)

        t = np.linspace(0, (self.N-1)*self.dt, self.N)
        
        # plot x
        ax_pos_x.plot(t[0:current_step+1], xg_world[0:current_step+1], '-g', marker='o', label='Human plan acceleration')
        ax_pos_x.plot(t[current_step:handover_point+1], xg_world[current_step:handover_point+1], '--g', marker='x', label='Human plan acceleration')        
        #ax_pos_x.plot(t[handover_point:self.N], xg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Human plan acceleration')        
    
        ax_pos_x.plot(t[0:current_step+1], xt_world[0:current_step+1], '-r', marker='o', label='Robot past plan velocity')
        ax_pos_x.plot(t[current_step:handover_point+1], xt_world[current_step:handover_point+1], '--r', marker='x', label='Robot planed velocity')
        #ax_pos_x.plot(t[handover_point:self.N], xt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Robot planed velocity')

        # Best handover location
        if handover_distance <= 0.1:
            ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        else:
            ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )

        #ax_pos_x.plot(t, final_xg_world, '--m', alpha=0.5, label='Final human velocity')
        #ax_pos_x.plot(t, final_xt_world, '--b', alpha=0.5, label='Final robot velocity')

        ax_pos_x.set_xlabel('t [s]')
        ax_pos_x.set_ylabel('ddX [m/s/s]')
    
        # plot y
        ax_pos_y.plot(t[0:current_step+1], yg_world[0:current_step+1], '-g', marker='o', label='Giver acceleration')
        ax_pos_y.plot(t[current_step:handover_point+1], yg_world[current_step:handover_point+1], '--g', marker='x', label='Giver plan')
        #ax_pos_y.plot(t[handover_point:self.N], yg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Giver plan (past handover)')
        
        ax_pos_y.plot(t[0:current_step+1], yt_world[0:current_step+1], '-r', marker='o', label='Taker acceleration')
        ax_pos_y.plot(t[current_step:handover_point+1], yt_world[current_step:handover_point+1], '--r', marker='x', label='Taker plan')
        #ax_pos_y.plot(t[handover_point:self.N], yt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Taker plan (past handover)')
      
        #ax_pos_y.plot(t, final_yg_world, '--m', alpha=0.5, label='Final human')
        #ax_pos_y.plot(t, final_yt_world, '--b', alpha=0.5, label='Final robot plan')
    
        # Best handover location
        #ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K = '+str( handover_distance ) )
        if handover_distance <= 0.1:
            ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + " = {:.2f}".format( handover_distance ) )
        else:
            ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + " = {:.2f}".format( handover_distance ) )
        
        ax_pos_y.set_xlabel('t [s]')
        ax_pos_y.set_ylabel('ddY [m/s/s]')

        # plot z
        ax_pos_zh.plot(t[0:current_step+1], zg_world[0:current_step+1], '-g', marker='o', label='Human True path')
        ax_pos_zh.plot(t[current_step:handover_point+1], zg_world[current_step:handover_point+1], '--g', marker='x', label='Human prediction')        
        #ax_pos_zh.plot(t[handover_point:self.N], zg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Human prediction')        
    
        #ax_pos_zh.plot(t, final_zg_world, '--m', alpha=0.5, label='Final human path')    

        ax_pos_zr.plot(t[0:current_step+1], zt_world[0:current_step+1], '-r', marker='o', label='Robot True path')
        ax_pos_zr.plot(t[current_step:handover_point+1], zt_world[current_step:handover_point+1], '--r', marker='x', label='Robot prediction')
        #ax_pos_zr.plot(t[handover_point:self.N], zt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Robot prediction')
    
        #ax_pos_zr.plot(t, final_zt_world, '--b', alpha=0.5, label='Final robot path')
    
        ax_pos_zh.set_xlabel('t [s]')
        ax_pos_zh.set_ylabel('ddZ [m/s/s]')
        ax_pos_zr.set_xlabel('t [s]')
        ax_pos_zr.set_ylabel('ddZ [m/s/s]')

        # Best handover location
        #ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K = '+str( handover_distance ) )
        #ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K = '+str( handover_distance ) )
        if handover_distance <= 0.1:
            #ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        else:
            #ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|r', linestyle='', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|r', linestyle='', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        

        # Title
        if world_frame:
            ax_pos_y.set_title('Handover acceleration in world frame, n = ' + str(current_step))
        else:
            ax_pos_y.set_title('Handover acceleration in shared frame, n = ' + str(current_step))
            

        # Legend
        ax_dict['legend'].set_visible(False)
        ax_dict['legend'].set_box_aspect(0.001)
        handles, labels = ax_pos_y.get_legend_handles_labels()
        fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=4)

        # Axis limits
        #ax_pos_x.set_ylim(0.0, 1.75)
        #ax_pos_y.set_ylim(-0.7, 0.7)
        #ax_pos_zh.set_ylim(-0.3, 0.4)
        #ax_pos_zr.set_ylim(-0.3, 0.4)

        ax_pos_x.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_y.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_zh.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_zr.set_xlim(-0.2, (self.N+1) * self.dt)

        # Window size (important for consistent figures)
        fig_pos.set_figheight(6)
        fig_pos.set_figwidth(12)

    def plot_one_handover(self, trajectory_idx, current_step=0, world_frame = True ):
        """
            Plot in worldframe, a trajectory given in shared frame
        """
        current_step = trajectory_idx # TODO
        # place recording better in the statemachine to get first plan

        #if self.robot_role == 'Giver':
        #    chest_Giver = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        #    chest_Taker = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )
        #elif self.robot_role == 'Taker':
        #    chest_Taker = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        #    chest_Giver = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )
        
        chest_Taker = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        chest_Giver = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )

        xg = self.x_giver_in_mid[trajectory_idx, :]
        yg = self.y_giver_in_mid[trajectory_idx, :]
        zg = self.z_giver_in_mid[trajectory_idx, :]
        xt = self.x_taker_in_mid[trajectory_idx, :]
        yt = self.y_taker_in_mid[trajectory_idx, :]
        zt = self.z_taker_in_mid[trajectory_idx, :]
        xg_world = np.zeros(xg.shape)
        yg_world = np.zeros(yg.shape)
        zg_world = np.zeros(zg.shape)
        xt_world = np.zeros(xt.shape)
        yt_world = np.zeros(yt.shape)
        zt_world = np.zeros(zt.shape)

        final_xg = self.x_true_giver_list
        final_yg = self.y_true_giver_list
        final_zg = self.z_true_giver_list
        final_xt = self.x_true_taker_list
        final_yt = self.y_true_taker_list
        final_zt = self.z_true_taker_list
        final_xg_world = np.zeros(xg.shape)
        final_yg_world = np.zeros(yg.shape)
        final_zg_world = np.zeros(zg.shape)
        final_xt_world = np.zeros(xt.shape)
        final_yt_world = np.zeros(yt.shape)
        final_zt_world = np.zeros(zt.shape)

        handover_point = self.predicted_handover_n_list[trajectory_idx]
        handover_distance = self.predicted_handover_K_list[trajectory_idx]

        if world_frame:
            # Transform
            for t in range(self.N):
                if t < self.no_trajectories:
                    TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
        
                P_n = np.matmul(TF, np.array([xg[t], yg[t], zg[t], 1]))  #giver point at time n
                xg_world[t] = P_n[0]
                yg_world[t] = P_n[1]
                zg_world[t] = P_n[2] # TF to shared frame

                P_n = np.matmul(TF, np.array([xt[t], yt[t], zt[t], 1]))  #taker point at time n
                xt_world[t] = P_n[0]
                yt_world[t] = P_n[1]
                zt_world[t] = P_n[2] # TF to shared frame

                #if t < self.M:
                P_n = np.matmul(TF, np.array([final_xg[t], final_yg[t], final_zg[t], 1]))  #final trajectory of handover, human
                final_xg_world[t] = P_n[0]
                final_yg_world[t] = P_n[1]
                final_zg_world[t] = P_n[2] # TF to shared frame

                P_n = np.matmul(TF, np.array([final_xt[t], final_yt[t], final_zt[t], 1]))  #final trajectory of handover, robot
                final_xt_world[t] = P_n[0]
                final_yt_world[t] = P_n[1]
                final_zt_world[t] = P_n[2] # TF to shared frame
        else:
            # Don't transform
            xg_world = xg
            yg_world = yg
            zg_world = zg
            xt_world = xt
            yt_world = yt
            zt_world = zt
            final_xg_world = final_xg
            final_yg_world = final_yg
            final_zg_world = final_zg
            final_xt_world = final_xt
            final_yt_world = final_yt
            final_zt_world = final_zt

        fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['legend',   'legend',   'legend'],
                                               ['legend',   'legend',   'legend'],
                                               ['legend',   'legend',   'legend']], layout='tight')

        ax_pos_x = ax_dict['x']
        ax_pos_y = ax_dict['y']
        ax_pos_zh = ax_dict['zh_plot']
        ax_pos_zr = ax_dict['zr_plot']
        ax_pos_x.grid(True)
        ax_pos_y.grid(True)
        ax_pos_zh.grid(True)
        ax_pos_zr.grid(True)

        t = np.linspace(0, (self.N-1)*self.dt, self.N)
        
        # plot x
        ax_pos_x.plot(t[0:current_step+1], xg_world[0:current_step+1], '-g', marker='o', label='Human True path')
        ax_pos_x.plot(t[current_step:handover_point+1], xg_world[current_step:handover_point+1], '--g', marker='x', label='Human ')        
        #ax_pos_x.plot(t[handover_point:self.N], xg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Human ')        
    
        ax_pos_x.plot(t[0:current_step+1], xt_world[0:current_step+1], '-r', marker='o', label='Robot True path')
        ax_pos_x.plot(t[current_step:handover_point+1], xt_world[current_step:handover_point+1], '--r', marker='x', label='Robot ')
        #ax_pos_x.plot(t[handover_point:self.N], xt_world[handover_point:self.N], '--r', alpha=0.5, marker='x', label='Robot ')

        # Best handover location
        if handover_distance <= 0.1:
            ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        else:
            ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )

        ax_pos_x.plot([t[n] for n in range(0, self.N)], final_xg_world[0:self.N], '--m', alpha=0.5, label='Final human path')
        ax_pos_x.plot([t[n] for n in range(0, self.N)], final_xt_world[0:self.N], '--b', alpha=0.5, label='Final robot path')

        ax_pos_x.set_xlabel('t [s]')
        ax_pos_x.set_ylabel('X [m]', loc='center')
    
        # plot y
        ax_pos_y.plot(t[0:current_step+1], yg_world[0:current_step+1], '-g', marker='o', label='Giver position')
        ax_pos_y.plot(t[current_step:handover_point+1], yg_world[current_step:handover_point+1], '--g', marker='x', label='Giver plan')
        #ax_pos_y.plot(t[handover_point:self.N], yg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Giver plan (past handover)')
        ax_pos_y.plot(t[0:current_step+1], yt_world[0:current_step+1], '-r', marker='o', label='Taker position')
        ax_pos_y.plot(t[current_step:handover_point+1], yt_world[current_step:handover_point+1], '--r', marker='x', label='Taker plan')
        #ax_pos_y.plot(t[handover_point:self.N], yt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Taker plan (past handover)')
      
        ax_pos_y.plot([t[n] for n in range(0, self.N)], final_yg_world[0:self.N], '--m', alpha=0.5, label='True giver position')
        ax_pos_y.plot([t[n] for n in range(0, self.N)], final_yt_world[0:self.N], '--b', alpha=0.5, label='True taker position')
    
        # Best handover location
        #ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K = '+str( handover_distance ) )
        if handover_distance <= 0.1:
            ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + " = {:.2f}".format( handover_distance ) )
        else:
            ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + " = {:.2f}".format( handover_distance ) )
        
        ax_pos_y.set_xlabel('t [s]')
        ax_pos_y.set_ylabel('Y [m]', loc='center')

        # plot z
        ax_pos_zh.plot(t[0:current_step+1], zg_world[0:current_step+1], '-g', marker='o', label='Human True path')
        ax_pos_zh.plot(t[current_step:handover_point+1], zg_world[current_step:handover_point+1], '--g', marker='x', label='Human prediction')        
        #ax_pos_zh.plot(t[handover_point:self.N], zg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Human prediction')        
    
        ax_pos_zh.plot([t[n] for n in range(0, self.N)], final_zg_world[0:self.N], '--m', alpha=0.5, label='Final human path')    

        ax_pos_zr.plot(t[0:current_step+1], zt_world[0:current_step+1], '-r', marker='o', label='Robot True path')
        ax_pos_zr.plot(t[current_step:handover_point+1], zt_world[current_step:handover_point+1], '--r', marker='x', label='Robot prediction')
        #ax_pos_zr.plot(t[handover_point:self.N], zt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Robot prediction')
    
        ax_pos_zr.plot([t[n] for n in range(0, self.N)], final_zt_world[0:self.N], '--b', alpha=0.5, label='Final robot path')
    
        ax_pos_zh.set_xlabel('t [s]')
        ax_pos_zh.set_ylabel('Z [m]', loc='center')
        ax_pos_zr.set_xlabel('t [s]')
        ax_pos_zr.set_ylabel('Z [m]', loc='center')

        # Best handover location
        #ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K = '+str( handover_distance ) )
        #ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K = '+str( handover_distance ) )
        if handover_distance <= 0.1:
            #ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        else:
            #ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|r', linestyle='', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|r', linestyle='', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        

        # Title
        if world_frame:
            ax_pos_y.set_title('Handover trajectory in world frame, n = ' + str(current_step) + ', Robot: ' + self.robot_role)
        else:
            ax_pos_y.set_title('Handover trajectory in shared frame, n = ' + str(current_step) + ', Robot: ' + self.robot_role)

        # Legend
        ax_dict['legend'].set_visible(False)
        ax_dict['legend'].set_box_aspect(0.001)
        handles, labels = ax_pos_y.get_legend_handles_labels()
        # Format for screen
        #fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=4)
        
        # Format for ikra paper
        fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=2)

        # Axis limits
        #ax_pos_x.set_ylim(0.0, 1.75)
        #ax_pos_y.set_ylim(-0.7, 0.7)
        #ax_pos_zh.set_ylim(-0.3, 0.4)
        #ax_pos_zr.set_ylim(-0.3, 0.4)

        ax_pos_x.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_y.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_zh.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_zr.set_xlim(-0.2, (self.N+1) * self.dt)

        # Window size (important for consistent figures)
        # Format for screen
        #fig_pos.set_figheight(6)
        #fig_pos.set_figwidth(12)
        
        # Format for ikra paper
        fig_pos.set_figheight(9)
        fig_pos.set_figwidth(10)


    def plot_one_handover2(self, trajectory_idx, current_step=0, world_frame = True ):
        """
            Plot in worldframe, a trajectory given in shared frame
            Use labels 'Robot' and 'Human'
        """
        current_step = trajectory_idx # TODO
        # place recording better in the statemachine to get first plan

        #if self.robot_role == 'Giver':
        #    chest_Giver = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        #    chest_Taker = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )
        #elif self.robot_role == 'Taker':
        #    chest_Taker = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        #    chest_Giver = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )
        
        chest_Taker = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        chest_Giver = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )

        xg = self.x_giver_in_mid[trajectory_idx, :]
        yg = self.y_giver_in_mid[trajectory_idx, :]
        zg = self.z_giver_in_mid[trajectory_idx, :]
        xt = self.x_taker_in_mid[trajectory_idx, :]
        yt = self.y_taker_in_mid[trajectory_idx, :]
        zt = self.z_taker_in_mid[trajectory_idx, :]
        xg_world = np.zeros(xg.shape)
        yg_world = np.zeros(yg.shape)
        zg_world = np.zeros(zg.shape)
        xt_world = np.zeros(xt.shape)
        yt_world = np.zeros(yt.shape)
        zt_world = np.zeros(zt.shape)

        final_xg = self.x_true_giver_list
        final_yg = self.y_true_giver_list
        final_zg = self.z_true_giver_list
        final_xt = self.x_true_taker_list
        final_yt = self.y_true_taker_list
        final_zt = self.z_true_taker_list
        final_xg_world = np.zeros(xg.shape)
        final_yg_world = np.zeros(yg.shape)
        final_zg_world = np.zeros(zg.shape)
        final_xt_world = np.zeros(xt.shape)
        final_yt_world = np.zeros(yt.shape)
        final_zt_world = np.zeros(zt.shape)

        handover_point = self.predicted_handover_n_list[trajectory_idx]
        handover_distance = self.predicted_handover_K_list[trajectory_idx]

        if world_frame:
            # Transform
            for t in range(self.N):
                if t < self.no_trajectories:
                    TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
        
                P_n = np.matmul(TF, np.array([xg[t], yg[t], zg[t], 1]))  #giver point at time n
                xg_world[t] = P_n[0]
                yg_world[t] = P_n[1]
                zg_world[t] = P_n[2] # TF to shared frame

                P_n = np.matmul(TF, np.array([xt[t], yt[t], zt[t], 1]))  #taker point at time n
                xt_world[t] = P_n[0]
                yt_world[t] = P_n[1]
                zt_world[t] = P_n[2] # TF to shared frame

                #if t < self.M:
                P_n = np.matmul(TF, np.array([final_xg[t], final_yg[t], final_zg[t], 1]))  #final trajectory of handover, human
                final_xg_world[t] = P_n[0]
                final_yg_world[t] = P_n[1]
                final_zg_world[t] = P_n[2] # TF to shared frame

                P_n = np.matmul(TF, np.array([final_xt[t], final_yt[t], final_zt[t], 1]))  #final trajectory of handover, robot
                final_xt_world[t] = P_n[0]
                final_yt_world[t] = P_n[1]
                final_zt_world[t] = P_n[2] # TF to shared frame
        else:
            # Don't transform
            xg_world = xg
            yg_world = yg
            zg_world = zg
            xt_world = xt
            yt_world = yt
            zt_world = zt
            final_xg_world = final_xg
            final_yg_world = final_yg
            final_zg_world = final_zg
            final_xt_world = final_xt
            final_yt_world = final_yt
            final_zt_world = final_zt

        fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zr_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['x',        'y',        'zh_plot'     ],
                                               ['legend',   'legend',   'legend'],
                                               ['legend',   'legend',   'legend'],
                                               ['legend',   'legend',   'legend']], layout='tight')

        ax_pos_x = ax_dict['x']
        ax_pos_y = ax_dict['y']
        ax_pos_zh = ax_dict['zh_plot']
        ax_pos_zr = ax_dict['zr_plot']
        ax_pos_x.grid(True)
        ax_pos_y.grid(True)
        ax_pos_zh.grid(True)
        ax_pos_zr.grid(True)

        t = np.linspace(0, (self.N-1)*self.dt, self.N)
        
        # plot x
        ax_pos_x.plot(t[0:current_step+1], xg_world[0:current_step+1], '-g', marker='o', label='Human True path')
        ax_pos_x.plot(t[current_step:handover_point+1], xg_world[current_step:handover_point+1], '--g', marker='x', label='Human ')        
        #ax_pos_x.plot(t[handover_point:self.N], xg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Human ')        
    
        ax_pos_x.plot(t[0:current_step+1], xt_world[0:current_step+1], '-r', marker='o', label='Robot True path')
        ax_pos_x.plot(t[current_step:handover_point+1], xt_world[current_step:handover_point+1], '--r', marker='x', label='Robot ')
        #ax_pos_x.plot(t[handover_point:self.N], xt_world[handover_point:self.N], '--r', alpha=0.5, marker='x', label='Robot ')

        # Best handover location
        if handover_distance <= 0.1:
            ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        else:
            ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )

        ax_pos_x.plot([t[n] for n in range(0, self.N)], final_xg_world[0:self.N], '--m', alpha=0.5, label='Final human path')
        ax_pos_x.plot([t[n] for n in range(0, self.N)], final_xt_world[0:self.N], '--b', alpha=0.5, label='Final robot path')

        ax_pos_x.set_xlabel('t [s]')
        ax_pos_x.set_ylabel('X [m]', loc='center')
    
        # plot y
        ax_pos_y.plot(t[0:current_step+1], yg_world[0:current_step+1], '-g', marker='o', label='Human position')
        ax_pos_y.plot(t[current_step:handover_point+1], yg_world[current_step:handover_point+1], '--g', marker='x', label='Human plan')
        #ax_pos_y.plot(t[handover_point:self.N], yg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Giver plan (past handover)')
        ax_pos_y.plot(t[0:current_step+1], yt_world[0:current_step+1], '-r', marker='o', label='Robot position')
        ax_pos_y.plot(t[current_step:handover_point+1], yt_world[current_step:handover_point+1], '--r', marker='x', label='Robot plan')
        #ax_pos_y.plot(t[handover_point:self.N], yt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Taker plan (past handover)')
      
        ax_pos_y.plot([t[n] for n in range(0, self.N)], final_yg_world[0:self.N], '--m', alpha=0.5, label='True human position')
        ax_pos_y.plot([t[n] for n in range(0, self.N)], final_yt_world[0:self.N], '--b', alpha=0.5, label='True robot position')
    
        # Best handover location
        #ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K = '+str( handover_distance ) )
        if handover_distance <= 0.1:
            ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + " = {:.2f}".format( handover_distance ) )
        else:
            ax_pos_y.plot( [t[handover_point], t[handover_point]], [yg_world[handover_point], yt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + " = {:.2f}".format( handover_distance ) )
        
        ax_pos_y.set_xlabel('t [s]')
        ax_pos_y.set_ylabel('Y [m]', loc='center')

        # plot z
        ax_pos_zh.plot(t[0:current_step+1], zg_world[0:current_step+1], '-g', marker='o', label='Human True path')
        ax_pos_zh.plot(t[current_step:handover_point+1], zg_world[current_step:handover_point+1], '--g', marker='x', label='Human prediction')        
        #ax_pos_zh.plot(t[handover_point:self.N], zg_world[handover_point:self.N], '--g', marker='x', alpha=0.5, label='Human prediction')        
    
        ax_pos_zh.plot([t[n] for n in range(0, self.N)], final_zg_world[0:self.N], '--m', alpha=0.5, label='Final human path')    

        ax_pos_zr.plot(t[0:current_step+1], zt_world[0:current_step+1], '-r', marker='o', label='Robot True path')
        ax_pos_zr.plot(t[current_step:handover_point+1], zt_world[current_step:handover_point+1], '--r', marker='x', label='Robot prediction')
        #ax_pos_zr.plot(t[handover_point:self.N], zt_world[handover_point:self.N], '--r', marker='x', alpha=0.5, label='Robot prediction')
    
        ax_pos_zr.plot([t[n] for n in range(0, self.N)], final_zt_world[0:self.N], '--b', alpha=0.5, label='Final robot path')
    
        ax_pos_zh.set_xlabel('t [s]')
        ax_pos_zh.set_ylabel('Z [m]', loc='center')
        ax_pos_zr.set_xlabel('t [s]')
        ax_pos_zr.set_ylabel('Z [m]', loc='center')

        # Best handover location
        #ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K = '+str( handover_distance ) )
        #ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K = '+str( handover_distance ) )
        if handover_distance <= 0.1:
            #ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|k', linestyle='--', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|k', linestyle='', label='Predicted handover, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        else:
            #ax_pos_x.plot( [t[handover_point], t[handover_point]], [xg_world[handover_point], xt_world[handover_point]], '|r', linestyle='--', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zh.plot( [t[handover_point], t[handover_point]], [zg_world[handover_point]+0.05, zg_world[handover_point]-0.05], '|r', linestyle='', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
            ax_pos_zr.plot( [t[handover_point], t[handover_point]], [zt_world[handover_point]+0.05, zt_world[handover_point]-0.05], '|r', linestyle='', label='Closest to, K' + str(int(handover_point)) + ' = '+str( handover_distance ) )
        

        # Title
        if world_frame:
            ax_pos_y.set_title('Handover trajectory in world frame, n = ' + str(current_step) + ', Robot: ' + self.robot_role)
        else:
            ax_pos_y.set_title('Handover trajectory in shared frame, n = ' + str(current_step) + ', Robot: ' + self.robot_role)

        # Legend
        ax_dict['legend'].set_visible(False)
        ax_dict['legend'].set_box_aspect(0.001)
        handles, labels = ax_pos_y.get_legend_handles_labels()
        # Format for screen
        #fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=4)
        
        # Format for ikra paper
        fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=2)

        # Axis limits
        #ax_pos_x.set_ylim(0.0, 1.75)
        #ax_pos_y.set_ylim(-0.7, 0.7)
        #ax_pos_zh.set_ylim(-0.3, 0.4)
        #ax_pos_zr.set_ylim(-0.3, 0.4)

        ax_pos_x.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_y.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_zh.set_xlim(-0.2, (self.N+1) * self.dt)
        ax_pos_zr.set_xlim(-0.2, (self.N+1) * self.dt)

        # Window size (important for consistent figures)
        # Format for screen
        #fig_pos.set_figheight(6)
        #fig_pos.set_figwidth(12)
        
        # Format for ikra paper
        fig_pos.set_figheight(9)
        fig_pos.set_figwidth(10)

# Functions for easy access
def plot_recorded_handovers(dt, robot_role, x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, 
                                            x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, 
                                            x_giver_chest_list, y_giver_chest_list, z_giver_chest_list,
                                            x_taker_chest_list, y_taker_chest_list, z_taker_chest_list,
                                            predicted_handover_n_list, predicted_handover_K_list):
    """
        Show plot of all handovers
        set robot_role as 'Giver' or 'Taker'
    """
    p = plotRecordedHandover(dt, robot_role)
    p.set_variables_from_lists(x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, x_giver_chest_list, y_giver_chest_list, z_giver_chest_list, x_taker_chest_list, y_taker_chest_list, z_taker_chest_list, predicted_handover_n_list, predicted_handover_K_list)
    p.plot_all_handovers(show_plots = True, save_plots = False)

def save_plot_recorded_handovers(dt, robot_role, 
                                     x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, 
                                     x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, 
                                     x_giver_chest_list, y_giver_chest_list, z_giver_chest_list,
                                     x_taker_chest_list, y_taker_chest_list, z_taker_chest_list,
                                     predicted_handover_n_list, predicted_handover_K_list,
                                     trajectory_list,
                                     dimensions,
                                     x_true_giver_list,
                                     y_true_giver_list,
                                     z_true_giver_list,
                                     x_true_taker_list,
                                     y_true_taker_list,
                                     z_true_taker_list
                                     ):
    p = plotRecordedHandover(dt, robot_role)
    p.set_variables_from_lists(x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, x_giver_chest_list, y_giver_chest_list, z_giver_chest_list, x_taker_chest_list, y_taker_chest_list, z_taker_chest_list, predicted_handover_n_list, predicted_handover_K_list, trajectory_list, dimensions,x_true_giver_list, y_true_giver_list, z_true_giver_list, x_true_taker_list, y_true_taker_list, z_true_taker_list)
    p.plot_all_handovers(show_plots = False, save_plots = True)


SHOW = False
SAVE = False

ROBOT_ROLE = 'Taker'

#mpl.rcParams['text.usetex'] = True



OPTIMIZE_ROBUSTNESS=True
QUANTITATIVE_OPTIMIZATION= True


# Output path for plots (if SAVE = True)
timestr = time.strftime("%Y%m%d-%H%M%S")
try:
    os.mkdir("NEW")
except OSError as error:
    print(error)    

SAVE_PATH = 'NEW/'
#PLOT_PATH = 'C:/Users/jfred/FlaoGDrive/Exjobb Drive folder/Exjobb/STL Generate Trajectory MILP/active-learn-stl-master/active-learn-stl-master/Plot_animation_2'
#PLOT_NAME = f'/RobotHumanROSbag_timestr_HumanGiver_RobotTaker_OptRho{OPTIMIZE_ROBUSTNESS:01}_QuOpt{QUANTITATIVE_OPTIMIZATION:01}_HardSTL{HARD_STL_CONSTRAINT:01}'

# Path for recorded handover data
RECORDED_DATA_PATH = 'G:/My Drive/Exjobb Drive folder/Exjobb/STL Generate Trajectory MILP/active-learn-stl-master/active-learn-stl-master/HumanRobotROSbag/RecordedHandover5/'
# RECORDED_DATA_PATH = 'C:/Users/jfred/FlaoGDrive/Exjobb Drive folder/Exjobb/ ...

#CONSTANTS
dimensions = ['giver_x',        'giver_y',      'giver_z',
              'giver_dx',       'giver_dy',     'giver_dz', 
              'giver_ddx',      'giver_ddy',    'giver_ddz', 
              'taker_x',        'taker_y',      'taker_z', 
              'taker_dx',       'taker_dy',     'taker_dz', 
              'taker_ddx',      'taker_ddy',    'taker_ddz', 
              'relative_x',     'relative_y',   'relative_z',
              'relative_dx',    'relative_dy',  'relative_dz',
              'relative_ddx',   'relative_ddy', 'relative_ddz',
              'K_p']
dt = 0.2 # [s] step time
print("dt = "+str(dt))

# Find idx for latest timestamp at sample frequency
sample_idx = lambda time_array : [idx for idx, val in enumerate( 
    [floor(time_array[i+1]/dt) - floor(time_array[i]/dt) for i in range(len(time_array)-1) ] ) if val != 0]

## LOAD CHEST TRACKING
#human_chest_data = pd.read_csv( 'pose_human_chest.csv' )
#human_chest_time_array = human_chest_data.rosbagTimestamp
#human_chest_time_array = (human_chest_time_array - human_chest_time_array[0])/1e9 # Convert to [s] relative to t_0
#human_chest_sample_idx = sample_idx( human_chest_time_array )
#human_chest_sample_idx.insert(0, 0)

#robot_chest = [0.152, 0, 0.359] # SHIFT IN X NEEDED< TOWARDS FRONT OF ROBOT > TODO

#if ROBOT_ROLE == 'Taker':
#    chest_Giver = np.array([human_chest_data.x[human_chest_sample_idx], human_chest_data.y[human_chest_sample_idx], human_chest_data.z[human_chest_sample_idx]]) # Human chest
#    chest_Taker = np.tile(robot_chest, (chest_Giver.shape[1], 1) ).T
#elif ROBOT_ROLE == 'Giver':
#    chest_Taker = np.array([human_chest_data.x[human_chest_sample_idx], human_chest_data.y[human_chest_sample_idx], human_chest_data.z[human_chest_sample_idx]]) # Human chest
#    chest_Giver = np.tile(robot_chest, (chest_Taker.shape[1], 1) ).T
#else:
#    print("Set ROBOT_ROLE = 'Taker' or 'Giver'!")
#    raise


## Load hand tracking

## Human hand
#human_data = pd.read_csv( 'pose_human_right_hand.csv' )
#human_time_array = human_data.rosbagTimestamp
#human_time_array = (human_time_array - human_time_array[0])/1e9 # Convert to [s] relative to t_0
#human_sample_idx = sample_idx( human_time_array )
#human_sample_idx.insert(0, 0)
## Downsample to dt
#x_h_n = [human_data.x[i] for i in human_sample_idx] # Still in map frame
#y_h_n = [human_data.y[i] for i in human_sample_idx]
#z_h_n = [human_data.z[i] for i in human_sample_idx]

## Robot hand
#robot_data = pd.read_csv( 'pose_robot_right_hand.csv' )
#robot_time_array = robot_data.rosbagTimestamp
#robot_time_array = (robot_time_array - robot_time_array[0])/1e9
#robot_sample_idx = sample_idx( robot_time_array )
#robot_sample_idx.insert(0, 0)
## Downsample to dt
#x_r_n = [robot_data.x[i] for i in robot_sample_idx] # Still in map frame
#y_r_n = [robot_data.y[i] for i in robot_sample_idx]
#z_r_n = [robot_data.z[i] for i in robot_sample_idx]

## TF the data!!!
#for i in range( chest_Giver.shape[1] ):
#    TF = points2tfmatrix(chest_Giver[:,i], chest_Taker[:,i]) # Transformation matrix at time 0
    
#    if i < len(x_h_n):
#        P_n = np.matmul(TF, np.array([x_h_n[i], y_h_n[i], z_h_n[i], 1]))  #human point at time n
#        x_h_n[i] = P_n[0]
#        y_h_n[i] = P_n[1]
#        z_h_n[i] = P_n[2] # TF to shared frame

#    if i < len(x_r_n):
#        P_n = np.matmul(TF, np.array([x_r_n[i], y_r_n[i], z_r_n[i], 1]))  #robot point at time n
#        x_r_n[i] = P_n[0]
#        y_r_n[i] = P_n[1]
#        z_r_n[i] = P_n[2] # TF to shared frame
## "Sim" time
#time_array = [i*dt for i in range(len(x_h_n))]

# List of STL parameters on the form
#   Psi_[t1, t2] (lb < var < ub)
#   where Psi is a temporal opperator, F or G. var is one of the variables defined in dimensions

# Choose STL specifications
# Human-like reach motion specification
STL100 = True       # Each predicate satisfies all validation data
STL95_100 = True    # Each predicate satisfies [95, 100) % of validation data
STL90_95 = True     # # Each predicate satisfies [90, 95) % of validation data

KSHIRSAGAR_APPROACH = True      # Use Kshirsagar inspired approach stratergy    F[0,t] (K_p < epsilon)
Kshirsagar_only = False         # Ignore STL_list, only use Kshirsagar approach
HARD_KSHIRSAGAR_CONSTRAINT = True # Robustness (pseudo-robustness) of Kshirsagar part must be positive
HARD_STL_CONSTRAINT= False      # Robustness (pseudo-robustness) of all STL must be positive at every step 

STL_list = []

if STL100:
    STL_list = STL_list + [
    # 100
    ["G", 0.223151178211, 0.708333282952, -2.46841433044, 'taker_ddx', 2.05264055303],
    ["F", 0.74969802979, 0.785963439897, -0.389260883251, 'giver_dx', 1.29409622911],
    ["F", 0.654249880171, 0.669201617548, -3.27329947235, 'taker_dy', 2.94348775323],
    ["F", 0.700452172496, 0.716844595274, -1.91186240993, 'giver_ddy', 1.2419525048],
    ["F", 0.976964547464, 1.01966006361, -0.645273347431, 'relative_dy', 0.520322341167],
    ["F", 0.58333317439, 1.30244911675, -0.396943465162, 'relative_x', -0.273916338011],
    ["F", 0.0, 0.666666696204, -0.326654981311, 'giver_x', -0.216408845105],
    ["F", 1.23623155955, 1.28331307505, -0.586599989971, 'giver_x', 0.297293921602],
    ["F", 0.0, 0.630137223623, -0.250292084993, 'giver_y', -0.139903574555],
    ["F", 0.5, 1.37500001797, -0.159802816159, 'giver_y', -0.0458131116289],
    ["F", 0.5, 1.08333337914, -0.166666758279, 'giver_y', -0.0504832135449],
    ]

if STL95_100:
    STL_list = STL_list + [
    # 95 - 100
    ["G", 1.7107324017e-06, 0.70166935848, -0.109689696874, 'giver_dz', 0.97520583434],
    ["G", 0.0421771293849, 0.535914360783, -0.0145996633283, 'giver_dz', 0.970700714394],
    ["G", 0.218788969978, 0.999999975415, -0.754180078693, 'taker_dx', 0.307376076286],
    ["G", 0.157462585123, 0.99725644927, -0.377553176086, 'taker_dz', 1.2913897368],
    ["G", 1.4531396314e-298, 0.581159658136, -0.783950500186, 'giver_ddx', 1.29984502429],
    ["G", 0.541712817846, 0.734860990249, -0.770429052475, 'giver_ddy', 0.773599294978],
    ["G", 0.735479751934, 1.28637742051, -0.960481696957, 'giver_ddy', 0.3860438672],
    ["G", 0.458777442996, 0.70821734877, -2.02582039871, 'giver_ddz', 0.564897307375],
    ["G", 0.209915256947, 0.919858869084, -2.14429595181, 'giver_ddz', 1.67691435153],
    ["G", 0.610146221852, 1.08333298309, -2.16807897701, 'giver_ddz', 0.725659941801],
    ["G", 0.223151178211, 0.708333282952, -2.46841433044, 'taker_ddx', 2.05264055303],
    ["G", 0.0640717623597, 0.541410945747, -0.0654871913186, 'relative_dx', 1.0608113033],
    ["G", 0.0626001188762, 1.0, -0.264471129591, 'relative_dy', 0.919144912321],
    ["G", 0.0114441023361, 0.583282838989, -0.642632724779, 'relative_dz', 0.761492438232],
    ["G", 0.458468605868, 0.53893737778, -1.38994144458, 'relative_ddx', 1.87728083267],
    ["G", 0.277701222439, 0.746200643766, -2.41908496538, 'relative_ddx', 2.52340687477],
    ["G", 0.792911668902, 0.86823189205, -2.76353221184, 'relative_ddx', 0.796355384544],
    ["G", 0.958334782591, 1.08079687465, -2.01014393052, 'relative_ddx', 0.497688448695],
    ["G", 0.291666659418, 0.61766479802, -1.67292748186, 'relative_ddy', 2.27841238892],
    ["G", 0.40073601284, 0.49999928726, -1.31090432364, 'relative_ddy', 2.1901510867],
    ["G", 0.880264002913, 1.16642107289, -2.24124095415, 'relative_ddy', 0.785947755525],
    ["G", 0.958333386617, 1.07331526291, -1.68181708447, 'relative_ddy', 0.227855365979],
    ["G", 0.387299525196, 0.541665414595, -3.15868882662, 'relative_ddz', 1.39923110409],
    ["G", 0.25000485272, 1.0, -4.310560464, 'relative_ddz', 3.05759090137],
    ["G", 1.00998178334, 1.24969928078, -1.18255397231, 'relative_ddz', 1.80091803966],
    ["G", 0.833604353145, 1.46579281064, -2.5730137431, 'relative_ddz', 2.80951798702],
    ["F", 0.0, 0.528297835644, 0.141330411726, 'giver_dx', 0.25901120066],
    ["F", 0.0, 0.489571724837, 0.142184826227, 'giver_dx', 0.260876222505],
    ["F", 0.521508699833, 1.08333340685, 0.0373344946545, 'giver_dx', 0.150486575364],
    ["F", 0.58333309461, 0.965937646168, -0.014424874789, 'giver_dz', 0.102564677179],
    ["F", 0.5, 1.34576459032, -0.0564553609465, 'taker_dx', 0.0568470673511],
    ["F", 0.617579245301, 1.19588980723, -0.0590855875727, 'taker_dz', 0.0590855849066],
    ["F", 0.0, 0.651192230233, 0.291645590143, 'giver_ddy', 0.416161579829],
    ["F", 0.389340084432, 0.41861182853, -1.16436196575, 'taker_ddy', 0.823368374555],
    ["F", 0.34262204389, 0.541666706974, -1.02525038341, 'taker_ddz', 1.75719037429],
    ["F", 0.487565837216, 0.54174036186, -2.14252600597, 'taker_ddz', 2.81461985968],
    ["F", 0.0, 0.592199963522, 0.341671526608, 'relative_dy', 0.461587011133],
    ["F", 0.5, 1.03359203152, 0.205399545692, 'relative_dy', 0.335920315227],
    ["F", 0.630687041993, 0.716691767036, -3.14387888955, 'relative_ddz', 2.03645460826],
    ["F", 1.16285555341, 1.20913827423, -0.805025879567, 'relative_ddz', 1.23174161066],
    ["F", 0.193252786943, 0.916666694284, -0.474593764612, 'relative_x', -0.363194552114],
    ["F", 0.833333311721, 1.18958202867, -0.387927058928, 'relative_x', -0.268659687293],
    ["F", 0.00970467308081, 0.750000002623, -0.323227914358, 'relative_y', -0.211789889328],
    ["F", 0.5, 1.10373117689, -0.207462353788, 'relative_y', -0.0892208172186],
    ["F", 0.5, 1.10480108456, -0.209602169114, 'relative_y', -0.084045389106],
    ["F", 0.5, 1.31108945083, -0.0521938982025, 'relative_z', 0.0666476495429],
    ["F", 0.541666615358, 1.04268863877, -0.0541782239085, 'relative_z', 0.0615931716434],
    ["G", 0.0588861035062, 0.49999997088, -0.490278442382, 'relative_y', -0.102879432216],
    ["G", 0.83335097368, 1.33290497543, -0.0656470807289, 'relative_z', 0.0742921804892],
    ["G", 0.833340223852, 1.2039592833, -0.0658688216422, 'relative_z', 0.0717780033921],
    ["F", 0.0, 0.666666696204, -0.326654981311, 'giver_x', -0.216408845105],
    ["F", 1.23623155955, 1.28331307505, -0.586599989971, 'giver_x', 0.297293921602],
    ["F", 0.0, 0.630137223623, -0.250292084993, 'giver_y', -0.139903574555],
    ["F", 0.5, 1.37500001797, -0.159802816159, 'giver_y', -0.0458131116289],
    ["F", 0.5, 1.08333337914, -0.166666758279, 'giver_y', -0.0504832135449],
    ["F", 0.0, 0.661974260085, 0.202319984413, 'taker_x', 0.313953476683],
    ["F", 0.79129152618, 1.25000000404, -0.0194501851173, 'taker_y', 0.106345705103],
    ["F", 0.0, 0.676183782105, -0.243517803743, 'taker_z', -0.13127603514],
    ["F", 0.879330045176, 0.91790128984, -0.402570339511, 'taker_z', 0.073038394786],
    ["G", 0.128459092682, 0.94003739376, 0.0832059588833, 'taker_x', 0.443461007521],
]

if STL90_95:
    STL_list = STL_list + [
    # 90 - 95
    ["G", 0.5, 1.49999990202, -0.0749625076818, 'giver_dx', 0.311918487176],
    ["G", 0.647279467082, 1.20781748261, -0.0769276574443, 'giver_dy', 0.366535537406],
    ["G", 0.167826326285, 0.623471384574, -0.715040159747, 'taker_dx', 0.0544025965605],
    ["G", 0.183354481021, 0.624999913526, -0.537423164091, 'taker_dy', 0.146280416817],
    ["G", 0.197377170932, 0.791665726551, -0.53086226606, 'taker_dy', 0.276190608221],
    ["G", 0.916666687815, 1.41627798659, -0.606712698493, 'giver_ddx', 0.288282266851],
    ["G", 0.725911114066, 1.08290398085, -0.855961545004, 'giver_ddx', 0.188842692702],
    ["G", 1.00781207623, 1.26901053013, -0.70221759261, 'giver_ddz', 0.407439109569],
    ["G", 0.46696923103, 0.502238732165, -1.05572111022, 'taker_ddy', 1.06283589353],
    ["G", 0.799180823411, 0.873611065282, -0.341262923399, 'taker_ddy', 1.34971528672],
    ["G", 0.0463151314924, 0.540312970837, -0.00824489049723, 'relative_dx', 1.0472599161],
    ["F", 0.583333311507, 0.816226320214, 0.163414727997, 'giver_dy', 0.279855763946],
    ["F", 0.541666675414, 0.836746276952, 0.164645582576, 'giver_dy', 0.280689523616],
    ["F", 0.5, 0.809560165797, 0.160656767, 'giver_dy', 0.276478014239],
    ["F", 0.0, 0.549080420128, -0.322426407675, 'taker_dx', -0.201110884542],
    ["F", 0.0, 0.567797259508, -0.20509247149, 'taker_dy', -0.0869745613812],
    ["F", 0.5, 1.02171722851, -0.0621817660496, 'taker_dy', 0.0584433511999],
    ["F", 0.5, 1.30461553579, -0.0590049894846, 'taker_ddy', 0.0590049883936],
    ["F", 0.0, 0.504333467486, 0.408510618472, 'relative_dx', 0.535648904902],
    ["F", 0.0, 0.521671465848, 0.427427762821, 'relative_dx', 0.555007791052],
    ["F", 0.849470350647, 0.889828549453, -0.389513797238, 'relative_dx', 0.423000968275],
    ["F", 0.0, 0.53499629575, 0.308083940808, 'relative_dy', 0.430348868804],
    ["F", 0.0, 0.541676336388, 0.0372816003484, 'relative_dz', 0.165264143663],
    ["F", 0.7822987741, 1.18757755556, -0.0591187249123, 'relative_dz', 0.0591187233279],
    ["F", 0.124999656215, 0.603453683157, -0.324899364366, 'relative_y', -0.21252037549],
    ["F", 0.0, 0.515492604153, -0.0528171923472, 'relative_z', 0.0686446172071],
    ["F", 0.0, 0.487433831467, -0.0535834076244, 'relative_z', 0.0695120017205],
    ["G", 0.291674764074, 0.536486935463, -0.655107988693, 'relative_x', -0.239546913309],
    ["G", 0.79314465484, 0.955295603026, -0.393088262699, 'relative_x', -0.267178802475],
    ["F", 0.624999995915, 1.06390854232, -0.233720649506, 'giver_x', -0.115638994749],
    ["F", 0.519489591549, 0.583333405875, -0.213542913108, 'giver_y', -0.0631889769589],
    ["G", 0.0942780982187, 0.868673240562, -0.367507344877, 'giver_x', -0.10925566737],
    ["G", 0.726637287455, 1.39204308172, -0.239375032052, 'giver_x', -0.0893696057889],
    ["G", 1.20833841352, 1.46487393746, -0.136761032203, 'giver_y', -0.00219269054168],
    ["G", 0.76631703502, 1.40742977065, -0.166328421385, 'giver_y', -0.00227791361718],
    ["G", 0.19307122218, 0.833333244357, -0.269748768172, 'giver_z', 0.0500028805929],
    ["F", 0.73386724126, 1.08333370055, 0.100401117081, 'taker_x', 0.228080772247],
    ["F", 0.499999962517, 0.91666803827, -0.0209407766901, 'taker_y', 0.103972101673],
    ["F", 0.207201400722, 0.833334294232, 0.00744934269999, 'taker_y', 0.122277200569],
    ["F", 0.603343914304, 1.00048219447, -0.0257122443711, 'taker_y', 0.10126030582],
    ["G", 0.685143618255, 1.44202567403, -0.0421143392063, 'taker_y', 0.124939658427],
    ["G", 0.883464615166, 1.2697072801, -0.0304310370439, 'taker_y', 0.121038597636],
]

max_time_in_spec = max( [ STL_list[j][2] for j in range(len(STL_list))] ) # Gets the maximum time of the handover


def parse_STL_list(STL_list, dt):
    # Takes STL_list and outputs an STLFormula
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

# Implementation of Kshirsagar approach, towards human
#   position only, no orientation
#   l1 norm
#   F[0, t] ( ||p_h - p_r - p_d|| < epsilon )

#   Parameters
#TODO: VERIFT THESE OFFSETS with real object and baxter holding them!
KSHIRSAGAR_DELTA = [0.3, 0.1, 0.0]# x,y,z offset of giver and taker hands on the object 
KSHIRSAGAR_EPSILON = 0.01
#KSHIRSAGAR_TIME = 2.0 # [s]
KSHIRSAGAR_TIME = max_time_in_spec

#   STL implementation
t_K = int(KSHIRSAGAR_TIME / dt) + 1
Spatial_predicate = STLFormula.Predicate('K_p', operatorclass.lt, KSHIRSAGAR_EPSILON, dimensions.index('K_p'))
Kshirsagar_phi = STLFormula.Eventually( Spatial_predicate, 0, t_K)
if KSHIRSAGAR_APPROACH == True:
    phi = STLFormula.Conjunction(phi, Kshirsagar_phi)

if Kshirsagar_only:
    phi = Kshirsagar_phi

# Remove negations from specification (not necesary, since specification is defined without negations)
phi_nnf = STLFormula.toNegationNormalForm(phi, False)

# Planner parameters
domain=[-10, 10] # Domain of signals
max_speed = 1 # m/s
max_giver_acceleration = [6.8506, 3.8406, 8.3751] # m/s^2 #  TODO: modify these parameters. maybe same for both giver and taker. maybe same in all directions?
max_taker_acceleration = [6.1479, 4.2769, 7.7459] # m/s^2
max_human_speed = 1 # m/s # NOT USED
max_change = [0,0,0, # speed limits, not used
                max_giver_acceleration[0] * dt, max_giver_acceleration[1] * dt, max_giver_acceleration[2] * dt, # Giver acceleration limits
                0,0,0, # speed limits, not used
                max_taker_acceleration[0] * dt, max_taker_acceleration[1] * dt, max_taker_acceleration[2] * dt] # Taker acceleration limits
U = [max_giver_acceleration[0], max_giver_acceleration[1], max_giver_acceleration[2],
     max_taker_acceleration[0], max_taker_acceleration[1], max_taker_acceleration[2]]


# Starting parameters # OVerwritten later, just sample values here
start=[-0.4604759871179124, -0.26378820211040827, -0.44995237273, 0.260456197785415, 0.2042333826402618, -0.449427643239]  
#if ROBOT_ROLE == 'Taker':
#    start=[x_h_n[0], y_h_n[0], z_h_n[0], # Giver position
#        x_r_n[0], y_r_n[0], z_r_n[0], # Taker position
#        ]
#else:
#    start=[x_r_n[0], y_r_n[0], z_r_n[0], # Giver position
#        x_h_n[0], y_h_n[0], z_h_n[0], # Taker position
#        ]
        

# Known start position
#start=[-0.386422400162729, -0.283712194922291, -0.279917642964524, 0.436710658871381, 0.174297651097749, -0.269005043964258]

# Initiate optimization problem for generating signals
problem = generate_signal_problem(phi, start, domain, dimensions, U, dt, 
                                    OPTIMIZE_ROBUSTNESS=OPTIMIZE_ROBUSTNESS,
                                    QUANTITATIVE_OPTIMIZATION= QUANTITATIVE_OPTIMIZATION,
                                    HARD_STL_CONSTRAINT= HARD_STL_CONSTRAINT, 
                                    KSHIRSAGAR_APPROACH= KSHIRSAGAR_APPROACH,
                                    KSHIRSAGAR_DELTA = KSHIRSAGAR_DELTA,
                                    Kshirsagar_phi = Kshirsagar_phi,
                                    HARD_KSHIRSAGAR_CONSTRAINT = HARD_KSHIRSAGAR_CONSTRAINT)

# Structure feedback / human motion prediction data
# Resample human data to dt sample time
#idx = [min(enumerate(human_data.time), key=lambda x: abs(target - x[1]))[0] for target in [x*dt for x in range(problem.phi.horizon * 2)]] # find closest time index!
#feedback = np.array([human_data.x_taker_RHand[idx], human_data.y_taker_RHand[idx], human_data.z_taker_RHand[idx]]).T
#true_giver = np.array([x_h_n, y_h_n, z_h_n]).T
#true_taker = feedback


# State machine
#   0: Idle
#   4: in pickup position
#   1: Reach phase
#   2: Handover failed (Reset)
#   3: Handover success (Reset)
state_machine = 0
trajectory_list=[]
n_list=[]
human_chest_list=[]
robot_chest_list=[]
x_human_chest_list=[]
y_human_chest_list=[]
z_human_chest_list=[]
x_robot_chest_list=[]
y_robot_chest_list=[]
z_robot_chest_list=[]

x_robot_in_mid=[]
y_robot_in_mid=[]
z_robot_in_mid=[]
x_human_in_mid=[]
y_human_in_mid=[]
z_human_in_mid=[]

x_true_human_list=[]
y_true_human_list=[]
z_true_human_list=[]
x_true_robot_list=[]
y_true_robot_list=[]
z_true_robot_list=[]

predicted_handover_n_list = []
predicted_handover_K_list = []

# Execution loop- How far in handvoer we are
n = 0

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('predict_robo_STL', anonymous=True)
    listener_1 = tf.TransformListener()
    listener_2 = tf.TransformListener()
    listener_3 = tf.TransformListener()
    listener_4 = tf.TransformListener()
    listener_5 = tf.TransformListener()
    listener_6 = tf.TransformListener()
    listener_7 = tf.TransformListener()
    if ROBOT_ROLE == 'Taker':
        robot_pos_idx = [dimensions.index('taker_x'), dimensions.index('taker_y'), dimensions.index('taker_z')]
        human_pos_idx = [dimensions.index('giver_x'), dimensions.index('giver_y'), dimensions.index('giver_z')]
        robot_vel_idx = [dimensions.index('taker_dx'), dimensions.index('taker_dy'), dimensions.index('taker_dz')]
        human_vel_idx = [dimensions.index('giver_dx'), dimensions.index('giver_dy'), dimensions.index('giver_dz')]
    elif ROBOT_ROLE == 'Giver':
        robot_pos_idx = [dimensions.index('giver_x'), dimensions.index('giver_y'), dimensions.index('giver_z')]
        human_pos_idx = [dimensions.index('taker_x'), dimensions.index('taker_y'), dimensions.index('taker_z')]
        robot_vel_idx = [dimensions.index('giver_dx'), dimensions.index('giver_dy'), dimensions.index('giver_dz')]
        human_vel_idx = [dimensions.index('taker_dx'), dimensions.index('taker_dy'), dimensions.index('taker_dz')]

    ### FLAG Set pick up position
    # Pick up position - To trgger handover. when human picks up the object. 
    #PK: making it in world frame now!. compare to world frame hand postion
    # The position of the object to pick up
    #i = int(16.6/dt) # picked from the graph!
    start_x = 1.023 #TODO VERIFY Translation: [1.214, 0.434, -0.177
    start_y = 0.352
    start_z = -0.159
    execution_time=0.0
    state_machine = 0


    # Execution loop- How far in handvoer we are
    n = 0
    ### FLAG Set thresholds
    Success_threshold = 0.1 # If K(t_n) < Success_threshold the handover is successfull.
    Enter_pickup_zone_threshold = 0.1 # Triggers move from state 0->4, 'idle'->'in pickup zone'
    Exit_pickup_zone_threshold = 0.1  # Triggers move from state 4->1, 'in pickup zone'->'reach phase'

    while not rospy.is_shutdown():
        try:
            rospy.sleep(0.2) # at 1000 HZ
            for t in range(0,2000): # Loop through starting points
                print('Exec time',execution_time)

            
                #pub_tf.publish(tfm)
                (trans_human_right_hand,rot) = listener_1.lookupTransform('world', 'Hand_right0', rospy.Time(0))
                #print("Right_hand_pos",trans)
                right_hand_pos = np.array(trans_human_right_hand, dtype='float32')
                #rospy.loginfo("Right_hand pos",trans) #TODO what's the correct format for logging variable value 
                (trans_robot_right_hand,rot2) = listener_2.lookupTransform('world', 'right_gripper', rospy.Time(0))
                #rospy.loginfo("Gripper_hand pos",trans2)    
                right_gripper_pos = np.array(trans_robot_right_hand, dtype='float32')

                (trans_human_right_hand_in_mid,rot6) = listener_6.lookupTransform('mid_point', 'Hand_right0', rospy.Time(0))
                #print("Right_hand_pos",trans)
                right_hand_in_mid_pos = np.array(trans_human_right_hand_in_mid, dtype='float32')
                #rospy.loginfo("Right_hand pos",trans) #TODO what's the correct format for logging variable value 
                (trans_robot_right_hand_in_mid,rot7) = listener_7.lookupTransform('mid_point', 'right_gripper', rospy.Time(0))
                #rospy.loginfo("Gripper_hand pos",trans2)    
                right_gripper_in_mid_pos = np.array(trans_robot_right_hand_in_mid, dtype='float32')
                

                (trans_human_chest,rot3) = listener_3.lookupTransform('world', 'Spine_Chest0', rospy.Time(0))
                #rospy.loginfo("Gripper_hand pos",trans2)    
                human_chest_pos = np.array(trans_human_chest, dtype='float32')
                
                (trans_robot_chest,rot4) = listener_4.lookupTransform('world', 'robot_chest', rospy.Time(0))
                #rospy.loginfo("Gripper_hand pos",trans2)    
                trans_robot_chest_pos = np.array(trans_robot_chest, dtype='float32')
                    #right_hand_human_in_mid = transform_to_pose(right_hand_in_mid_pos)
                    #pub_human_hand_in_mid.publish(right_hand_human_in_mid)

                #right_gripper_in_mid = transform_to_pose(right_gripper_in_mid_pos)
                #pub_robot_hand_in_mid.publish(right_gripper_in_mid)    


                if state_machine == 0: # Idle
                    # Set variables
                    print('SMC=',state_machine)
                    t_0 = t # Start of handover set to current time
                    n = 0 # Index of the current step of execution

                    ## Set/fix current human position values (Recorded)
                    #problem.unfix_variables([n], human_pos_idx)
                    #problem.fix_variables([n], human_pos_idx, [[right_hand_in_mid_pos[0], right_hand_in_mid_pos[1], right_hand_in_mid_pos[2]]]) # real postion in mid frame. need updation.
                    ## Set/fix robot current position (Recorded)
                    #problem.unfix_variables([n], robot_pos_idx)
                    #problem.fix_variables([n], robot_pos_idx, [[right_gripper_in_mid_pos[0], right_gripper_in_mid_pos[1], right_gripper_in_mid_pos[2]]])

                    ### NEW
                    # update past states s[-2]=s[-1]
                    # update past states s[-1]=s[0]
                    # set current state  s[0]=current position        
                    if ROBOT_ROLE == 'Taker':
                        current_giver_shared = [right_hand_in_mid_pos[0], right_hand_in_mid_pos[1], right_hand_in_mid_pos[2]] # current position of [human_x, human_y, human_z]
                        current_taker_shared = [right_gripper_in_mid_pos[0], right_gripper_in_mid_pos[1], right_gripper_in_mid_pos[2]] # current position of [robot_x, robot_y, robot_z]
                    elif ROBOT_ROLE == 'Giver':
                        current_giver_shared = [right_gripper_in_mid_pos[0], right_gripper_in_mid_pos[1], right_gripper_in_mid_pos[2]]
                        current_taker_shared = [right_hand_in_mid_pos[0], right_hand_in_mid_pos[1], right_hand_in_mid_pos[2]]
                    problem.set_leading_states(current_giver_shared, current_taker_shared)

                    # Execute planner at t_n = 0
                    start_execution_time = time.monotonic()
                    solution_found = problem.generate_path(n)
                    execution_time = time.monotonic() - start_execution_time
                    trajectory = problem.get_path()

                    
                    
                    # eucl distance from pickup start position  
                    # PK means the hand is now in the object pick zone # COORDINATES NOW IN WORLD FRAME FOR THIS
                    distance_hand=sqrt((right_hand_pos[0]-start_x)**2 + (right_hand_pos[1]-start_y)**2 + (right_hand_pos[2]-start_z)**2)
                    #enter_pickup_position = 0.05 < sqrt((right_hand_pos[0]-start_x)**2 + (right_hand_pos[1]-start_y)**2 + (right_hand_pos[2]-start_z)**2)

                    #print(sqrt((right_hand_pos[0]-start_x)**2 + (right_hand_pos[1]-start_y)**2 + (right_hand_pos[2]-start_z)**2))
                    # if enter_pickup_position:
                    #     state_machine = 4
                    if distance_hand < Enter_pickup_zone_threshold:
                        state_machine = 4    

                elif state_machine == 4: # in pickup position
                    print("In pickup zone, SMC=",state_machine)
                    # Set variables
                    t_0 = t # Start of handover set to current time
                    n = 0 # Index of the current step of execution

                    ## Set/fix current human position values (Recorded)
                    #problem.unfix_variables([n], human_pos_idx)
                    #problem.fix_variables([n], human_pos_idx, [[right_hand_in_mid_pos[0], right_hand_in_mid_pos[1], right_hand_in_mid_pos[2]]]) # real postion in mid frame. need updation.
                    ## Set/fix robot current position (Recorded)
                    #problem.unfix_variables([n], robot_pos_idx)
                    #problem.fix_variables([n], robot_pos_idx, [[right_gripper_in_mid_pos[0], right_gripper_in_mid_pos[1], right_gripper_in_mid_pos[2]]])

                    ### NEW
                    # update past states s[-2]=s[-1]
                    # update past states s[-1]=s[0]
                    # set current state  s[0]=current position        
                    if ROBOT_ROLE == 'Taker':
                        current_giver_shared = [right_hand_in_mid_pos[0], right_hand_in_mid_pos[1], right_hand_in_mid_pos[2]] # current position of [human_x, human_y, human_z]
                        current_taker_shared = [right_gripper_in_mid_pos[0], right_gripper_in_mid_pos[1], right_gripper_in_mid_pos[2]] # current position of [robot_x, robot_y, robot_z]
                    elif ROBOT_ROLE == 'Giver':
                        current_giver_shared = [right_gripper_in_mid_pos[0], right_gripper_in_mid_pos[1], right_gripper_in_mid_pos[2]]
                        current_taker_shared = [right_hand_in_mid_pos[0], right_hand_in_mid_pos[1], right_hand_in_mid_pos[2]]
                    problem.set_leading_states(current_giver_shared, current_taker_shared)

                    # Execute planner at t_n = 0
                    start_execution_time = time.monotonic()
                    solution_found = problem.generate_path(n)
                    execution_time = time.monotonic() - start_execution_time
                    trajectory = problem.get_path()

                    if ROBOT_ROLE == 'Taker':
                        robot_next_position = problem.get_taker_plan(n+1)
                    if ROBOT_ROLE == 'Giver':
                        robot_next_position = problem.get_giver_plan(n+1)

                    #GET ALL THE PLANNED PATH OF ROBOT, publish over a topic:
                    TF = points2invtfmatrix(human_chest_pos, trans_robot_chest_pos)
        
                    robot_next_position_world = np.matmul(TF, np.array([robot_next_position[0], robot_next_position[1], robot_next_position[2], 1]))  #giver point at time n
                    #robot_next_position_world[0] = P_n[0]
                    #robot_next_position_world[1] = P_n[1]
                    #robot_next_position_world[2] = P_n[2] # TF to shared frame
                    tf_Rob = geometry_msgs.msg.TransformStamped()
                    tf_Rob.header.frame_id = "world"
                    tf_Rob.header.stamp = rospy.Time.now()
                    tf_Rob.child_frame_id = "robot_desired_pose"
                    #CHECK!
                    tf_Rob.transform.translation.x = robot_next_position_world[0]
                    tf_Rob.transform.translation.y = robot_next_position_world[1]
                    tf_Rob.transform.translation.z = robot_next_position_world[2]
                    tf_Rob.transform.rotation.x = 0
                    tf_Rob.transform.rotation.y = 0
                    tf_Rob.transform.rotation.z = 0
                    tf_Rob.transform.rotation.w = 1

                    #tfm = tf.msg.tfMessage([t])
                    #broadcaster_robot_desired_pose.sendTransform(tf_Rob)
                    print(robot_next_position_world)
                    print(tf_Rob)

                    # # Check starting conditions
                    # #starting_condition = (x_h_n[t] > -0.3)  # To be implemented
                    #starting_condition = ( (t == int(7/dt)) or (t == int(11/dt)) or (t == int(15/dt)) or (t == int(20/dt)) or (t == int(25/dt)) ) # (Identified manually at)
                    #starting_condition = (t == int(6/dt) or t==int(17/dt) or t==int(27/dt) or t==int(38/dt)) 
                    #starting_condition = (t == int(18/dt))
                    #starting_condition = t == int(38.6/dt) or t==int(45.6/dt) # t==int(27.4/dt)
                    
                    # eucl distance from pickup start position # LEAVING THE PICKUP ZONE
                    #starting_condition = 0.1 > sqrt((right_hand_pos[0]-start_x)**2 + (right_hand_pos[1]-start_y)**2 + (right_hand_pos[2]-start_z)**2)
                    distance_hand=sqrt((right_hand_pos[0]-start_x)**2 + (right_hand_pos[1]-start_y)**2 + (right_hand_pos[2]-start_z)**2)
                    
                    #print('Distance is now ',sqrt((right_hand_pos[0]-start_x)**2 + (right_hand_pos[1]-start_y)**2 + (right_hand_pos[2]-start_z)**2))
                    
                    #if starting_condition:
                    if distance_hand > Exit_pickup_zone_threshold:
                        state_machine = 1

                        ### NEW
                        ### TODO use outputs of plan
                        # Get controll output
                        if ROBOT_ROLE == 'Taker':
                            robot_next_position = problem.get_taker_plan(n+1)
                            human_next_position = problem.get_giver_plan(n+1)
                        if ROBOT_ROLE == 'Giver':
                            robot_next_position = problem.get_giver_plan(n+1)
                            human_next_position = problem.get_taker_plan(n+1)

                        #GET ALL THE PLANNED PATH OF ROBOT, publish over a topic:
                        TF = points2invtfmatrix(human_chest_pos, trans_robot_chest_pos)
            
                        robot_next_position_world = np.matmul(TF, np.array([robot_next_position[0], robot_next_position[1], robot_next_position[2], 1]))  #giver point at time n
                        #robot_next_position_world[0] = P_n[0]
                        #robot_next_position_world[1] = P_n[1]
                        #robot_next_position_world[2] = P_n[2] # TF to shared frame
                        tf_Rob = geometry_msgs.msg.TransformStamped()
                        tf_Rob.header.frame_id = "world"
                        tf_Rob.header.stamp = rospy.Time.now()
                        tf_Rob.child_frame_id = "robot_desired_pose"
                        #CHECK!
                        tf_Rob.transform.translation.x = robot_next_position_world[0]
                        tf_Rob.transform.translation.y = robot_next_position_world[1]
                        tf_Rob.transform.translation.z = robot_next_position_world[2]
                        tf_Rob.transform.rotation.x = 0
                        tf_Rob.transform.rotation.y = 0
                        tf_Rob.transform.rotation.z = 0
                        tf_Rob.transform.rotation.w = 1

                        #tfm = tf.msg.tfMessage([t])
                        broadcaster_robot_desired_pose.sendTransform(tf_Rob)
                        print(robot_next_position_world)
                        print(tf_Rob)
                        trajectory_list.append(trajectory)
                        human_chest_list.append(human_chest_pos)
                        robot_chest_list.append(trans_robot_chest_pos)
                        x_human_chest_list.append(human_chest_pos[0])
                        y_human_chest_list.append(human_chest_pos[1])
                        z_human_chest_list.append(human_chest_pos[2])
                        x_robot_chest_list.append(trans_robot_chest_pos[0])
                        y_robot_chest_list.append(trans_robot_chest_pos[1])
                        z_robot_chest_list.append(trans_robot_chest_pos[2])
                        x_robot_in_mid.append( [ trajectory[a][robot_pos_idx[0]] for a in range(0,phi.horizon+1) ] )
                        y_robot_in_mid.append( [ trajectory[a][robot_pos_idx[1]] for a in range(0,phi.horizon+1) ] )
                        z_robot_in_mid.append( [ trajectory[a][robot_pos_idx[2]] for a in range(0,phi.horizon+1) ] )
                        x_human_in_mid.append( [ trajectory[a][human_pos_idx[0]] for a in range(0,phi.horizon+1) ] )
                        y_human_in_mid.append( [ trajectory[a][human_pos_idx[1]] for a in range(0,phi.horizon+1) ] )
                        z_human_in_mid.append( [ trajectory[a][human_pos_idx[2]] for a in range(0,phi.horizon+1) ] )
                        n_handover = problem.get_predicted_handover_step( Success_threshold, n )
                        predicted_handover_n_list.append( n_handover )
                        predicted_handover_K_list.append( problem.get_K( n_handover ) )
                        
                        x_true_human_list.append(right_hand_in_mid_pos[0])
                        y_true_human_list.append(right_hand_in_mid_pos[1])
                        z_true_human_list.append(right_hand_in_mid_pos[2])
                        x_true_robot_list.append(right_gripper_in_mid_pos[0])
                        y_true_robot_list.append(right_gripper_in_mid_pos[1])
                        z_true_robot_list.append(right_gripper_in_mid_pos[2])

                        if (SAVE or SHOW):
                            pass
                            #plot_trajectory2(phi_nnf, n , trajectory , dt , ROBOT_ROLE , close_flag=False)
                        if SAVE:
                            pass
                            #plt.savefig(SAVE_PATH + f'tstart{t_0:}_n{n:}_t{t:}.png')
                        if SHOW:
                            pass
                            #plt.show()
                        #plt.close()

                elif state_machine == 1: # Reach phase
                    print('Reach phase , SMC=',state_machine)
                    # Set variables
                    n += 1 

                    if UPDATE_HUMAN_FROM_PLAN:
                        problem.fix_variables([n], human_pos_idx, [human_next_position])
                    else:
                        # Set/fix current human position values (Recorded)
                        problem.fix_variables([n], human_pos_idx, [[right_hand_in_mid_pos[0], right_hand_in_mid_pos[1], right_hand_in_mid_pos[2]]])
                    # Fix robot to previous control output 
                    #   R_{t_n} <-- \hat{R}^{t_{n-1}}_{t_n}
                    #   Where \hat{R}^{t_{n-1}}_{t_n} is the robots predicted position at t_n, calculated at t_{n-1}
                    #   and R_{t_n} is the true position

                    ### FLAG Not using true position to update robot position in plan
                    
                    if UPDATE_ROBOT_FROM_PLAN:
                        ### USE PREDICTED POSITION
                        problem.fix_variables([n], robot_pos_idx, [robot_next_position]) ### Predicted position
                    else:
                        ### USE REAL ROBOT POSITION
                        problem.fix_variables([n], robot_pos_idx, [[right_gripper_in_mid_pos[0], right_gripper_in_mid_pos[1], right_gripper_in_mid_pos[2]]]) ### Real position

                    # Execute planner at t_n = n
                    start_execution_time = time.monotonic()
                    solution_found = problem.generate_path(n)
                    execution_time = time.monotonic() - start_execution_time
                    trajectory = problem.get_path()
                    
                    ### NEW
                    ### TODO use outputs of plan
                    # Get controll output
                    if ROBOT_ROLE == 'Taker':
                        robot_next_position = problem.get_taker_plan(n+1)
                        human_next_position = problem.get_giver_plan(n+1)
                    if ROBOT_ROLE == 'Giver':
                        robot_next_position = problem.get_giver_plan(n+1)
                        human_next_position = problem.get_taker_plan(n+1)
                    
                    #GET ALL THE PLANNED PATH OF ROBOT, publish over a topic:
                    TF = points2invtfmatrix(human_chest_pos, trans_robot_chest_pos)
        
                    robot_next_position_world = np.matmul(TF, np.array([robot_next_position[0], robot_next_position[1], robot_next_position[2], 1]))  #giver point at time n
                    #robot_next_position_world[0] = P_n[0]
                    #robot_next_position_world[1] = P_n[1]
                    #robot_next_position_world[2] = P_n[2] # TF to shared frame
                    tf_Rob = geometry_msgs.msg.TransformStamped()
                    tf_Rob.header.frame_id = "world"
                    tf_Rob.header.stamp = rospy.Time.now()
                    tf_Rob.child_frame_id = "robot_desired_pose"
                    #CHECK!
                    tf_Rob.transform.translation.x = robot_next_position_world[0]
                    tf_Rob.transform.translation.y = robot_next_position_world[1]
                    tf_Rob.transform.translation.z = robot_next_position_world[2]
                    tf_Rob.transform.rotation.x = 0
                    tf_Rob.transform.rotation.y = 0
                    tf_Rob.transform.rotation.z = 0
                    tf_Rob.transform.rotation.w = 1

                    #tfm = tf.msg.tfMessage([t])
                    broadcaster_robot_desired_pose.sendTransform(tf_Rob)
                    print(robot_next_position_world)
                    print(tf_Rob)
                                        

                    trajectory_list.append(trajectory)
                    human_chest_list.append(human_chest_pos)
                    robot_chest_list.append(trans_robot_chest_pos)
                    x_human_chest_list.append(human_chest_pos[0])
                    y_human_chest_list.append(human_chest_pos[1])
                    z_human_chest_list.append(human_chest_pos[2])
                    x_robot_chest_list.append(trans_robot_chest_pos[0])
                    y_robot_chest_list.append(trans_robot_chest_pos[1])
                    z_robot_chest_list.append(trans_robot_chest_pos[2])
                    x_robot_in_mid.append( [ trajectory[a][robot_pos_idx[0]] for a in range(0,phi.horizon+1) ] )
                    y_robot_in_mid.append( [ trajectory[a][robot_pos_idx[1]] for a in range(0,phi.horizon+1) ] )
                    z_robot_in_mid.append( [ trajectory[a][robot_pos_idx[2]] for a in range(0,phi.horizon+1) ] )
                    x_human_in_mid.append( [ trajectory[a][human_pos_idx[0]] for a in range(0,phi.horizon+1) ] )
                    y_human_in_mid.append( [ trajectory[a][human_pos_idx[1]] for a in range(0,phi.horizon+1) ] )
                    z_human_in_mid.append( [ trajectory[a][human_pos_idx[2]] for a in range(0,phi.horizon+1) ] )
                    n_handover = problem.get_predicted_handover_step( Success_threshold, n )
                    predicted_handover_n_list.append( n_handover )
                    predicted_handover_K_list.append( problem.get_K( n_handover ) )
                    
                    x_true_human_list.append(right_hand_in_mid_pos[0])
                    y_true_human_list.append(right_hand_in_mid_pos[1])
                    z_true_human_list.append(right_hand_in_mid_pos[2])
                    x_true_robot_list.append(right_gripper_in_mid_pos[0])
                    y_true_robot_list.append(right_gripper_in_mid_pos[1])
                    z_true_robot_list.append(right_gripper_in_mid_pos[2])

                    n_list.append(n)
                    print(f'K = {problem.get_K(n):}')
                    if (SAVE or SHOW):
                        pass
                        #plot_trajectory2(phi_nnf, n , trajectory , dt , ROBOT_ROLE , close_flag=False, true_human_x = [x_h_n[t_0 + i] for i in range(phi.horizon+1)], true_human_y = [y_h_n[t_0 + i] for i in range(phi.horizon+1)], true_human_zG = [z_h_n[t_0 + i] for i in range(phi.horizon+1)] )
                        plot_trajectory2(phi_nnf, n , trajectory , dt , ROBOT_ROLE , close_flag=False)
                    if SAVE:
                        pass
                        plt.savefig(SAVE_PATH + f'tstart{t_0:}_n{n:}_t{t:}.png' )
                    if SHOW:
                        pass
                        #plt.show()
                    plt.close()

                    # Check if handover timeout
                    if n >= phi.horizon -1: 
                        print('Handover timeout')
                        state_machine = 2 # Fail
                        # 3 sec horizon. if not handover till then, it fails..
                    # Check if handover failed
                    fail_condition = not solution_found # To be decided
                    if fail_condition:
                        print('Handover failed, planner could not find a solution')
                        # Check why
                        # How high is max acceleration?
                        # How high is max speed?
                        max_acc = problem.get_max_acceleration([a for a in range(n+1)], human_pos_idx)

                        state_machine = 2 # Fail (Reset state)

                    # Check if handover succeeded # VERIFY THIS> IF this is real or predicted?!
                    if problem.get_K(n) < Success_threshold: # KSHIRSAGAR_EPSILON:
                        state_machine = 3 # Success (Reset state)

                 # Reset states
                elif state_machine == 2 or state_machine == 3:
                    n += 1
                    if state_machine == 2:
                        print('handover failed')
                        #break
                    if state_machine == 3:
                        print('handover succeeded')
                        #break

                    # Log true positions after handover
                    x_true_human_list.append( right_hand_in_mid_pos[0] )
                    y_true_human_list.append( right_hand_in_mid_pos[1] )
                    z_true_human_list.append( right_hand_in_mid_pos[2] )
                    x_true_robot_list.append( right_gripper_in_mid_pos[0] )
                    y_true_robot_list.append( right_gripper_in_mid_pos[1] )
                    z_true_robot_list.append( right_gripper_in_mid_pos[2] )
        
                    # Exit planner after handover window
                    if n > phi.horizon:
                        break
                    
                    #problem.unfix_variables( [a for a in range(phi.horizon)], [a for a in range(len( dimensions ))] )

                    #state_machine = 0 # Idle

                print(f't = {t:}*dt = {t*dt:}; t_0 = t_{t_0:} = {t_0*dt:}; n = {n:}; state = {state_machine:}')
                
                t=t+1
                
            print('timeout!')        
            #pose_robot_right_hand_in_mid
            
            
            
            

            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue 

    # Plot with robot as taker
    save_plot_recorded_handovers(dt, ROBOT_ROLE, 
                             x_human_in_mid, 
                             y_human_in_mid, 
                             z_human_in_mid, 
                             x_robot_in_mid, 
                             y_robot_in_mid, 
                             z_robot_in_mid, 
                             x_human_chest_list, 
                             y_human_chest_list, 
                             z_human_chest_list,
                             x_robot_chest_list, 
                             y_robot_chest_list, 
                             z_robot_chest_list,
                             predicted_handover_n_list,
                             predicted_handover_K_list,
                             trajectory_list,
                             dimensions,
                             x_true_human_list,
                             y_true_human_list,
                             z_true_human_list,
                             x_true_robot_list,
                             y_true_robot_list,
                             z_true_robot_list
                             )


    for i in range(0,len(n_list)):
        #plot_trajectory2(phi_nnf, n_list[i] , trajectory_list[i] , dt , ROBOT_ROLE , close_flag=False)

        #plt.savefig("NEW/" + f'tstart{t_0:}_n{n_list[i]:}_t{t:}.png')
        #plot_recorded_handovers(dt, ROBOT_ROLE, human_chest_list,x_human_in_mid[i], y_human_in_mid[i], z_human_in_mid[i],x_robot_in_mid[i], y_robot_in_mid[i] , z_robot_in_mid[i] ,x_human_chest_list[i] , y_human_chest_list[i] ,z_human_chest_list[i] ,x_robot_chest_list[i] ,y_robot_chest_list[i] ,z_robot_chest_list[i] )

        pass
        
        #plot_in_worldframe(x_human_in_mid, y_human_in_mid, z_human_in_mid, x_robot_in_mid, y_robot_in_mid, z_robot_in_mid, human_chest_list, robot_chest_list, 0.2, n_list, true_human_x=False, true_human_y=False, true_human_zG=False, true_human_zT=False)
        #plt.savefig(SAVE_PATH + f'World_tstart{t_0:}_n{n_list[i]:}_t{t:}.png')
    with open('NEW/x_human_in_mid.csv', 'w') as f:
        for d in x_human_in_mid:
            f.write(str(d))
            f.write("\n")

    with open('NEW/y_human_in_mid.csv', 'w') as f:
        for d in y_human_in_mid:
            f.write(str(d))
            f.write("\n")

    with open('NEW/z_human_in_mid.csv', 'w') as f:
        for d in z_human_in_mid:
            f.write(str(d))
            f.write("\n")

    with open('NEW/x_rob_in_mid.csv', 'w') as f:
        for d in x_robot_in_mid:
            f.write(str(d))
            f.write("\n")

    with open('NEW/y_rob_in_mid.csv', 'w') as f:
        for d in y_robot_in_mid:
            f.write(str(d))
            f.write("\n")

    with open('NEW/z_rob_in_mid.csv', 'w') as f:
        for d in z_robot_in_mid:
            f.write(str(d))
            f.write("\n")

    with open('NEW/human_chest_list.csv', 'w') as f:
        for d in human_chest_list:
            f.write(str(d))
            f.write("\n")

    with open('NEW/robot_chest_list.csv', 'w') as f:
        for d in robot_chest_list:
            f.write(str(d))
            f.write("\n")
    #plt.show()

    #rospy.Subscriber("/body_tracking_data", MarkerArray, callback)

    #rosrun tf tf_echo world Hand_right0

    # static_transformStamped.transform.translation.x=1
    # static_transformStamped.transform.translation.y=2
    # static_transformStamped.transform.translation.z=3
    
    # static_transformStamped.transform.rotation.x=0
    # static_transformStamped.transform.rotation.y=0
    # static_transformStamped.transform.rotation.z=0
    
    # static_transformStamped.transform.rotation.w=1
    
    # spin() simply keeps python from exiting until this node is stopped
    #rospy.spin()

if __name__ == '__main__':
   listener()




