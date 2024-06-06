#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from tfmatrix import points2tfmatrix, points2invtfmatrix
import pandas as pd # pandas==2.0.3
import os

"""
asdf 
"""
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
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

            # Comment/uncomment to produce x_position plot in world frame 
            self.plot_x_position(trajectory_idx, world_frame = True)
            if save_plots:
                #plt.show(block=True)
                plt.savefig( save_location + 'world_frame/' + 'plan_world_frame_x_pos' + str(trajectory_idx) + '.png' )
                plt.close()

            # Comment/uncomment to produce position plot in world frame 
            self.plot_one_handover(trajectory_idx, world_frame = True)
            if save_plots:
                #plt.show(block=True)
                plt.savefig( save_location + 'world_frame/' + 'plan_world_frame_pos' + str(trajectory_idx) + '.png' )
                plt.close()

            # Comment/uncomment to produce velocity plot in world frame 
            self.plot_one_handover_velocity(trajectory_idx, world_frame = True)
            if save_plots:
                plt.savefig( save_location + 'world_frame/' + 'plan_world_frame_vel' + str(trajectory_idx) + '.png' )
                plt.close()

            # Comment/uncomment to produce acceleration plot in world frame 
            self.plot_one_handover_acceleration(trajectory_idx, world_frame = True)
            if save_plots:
                plt.savefig( save_location + 'world_frame/' + 'plan_world_frame_acc' + str(trajectory_idx) + '.png' )
                plt.close()

            self.plot_one_handover(trajectory_idx, world_frame = False)
            if save_plots:
                plt.savefig( save_location + 'shared_frame/' + 'plan_share_frame_pos' + str(trajectory_idx) + '.png' )
                plt.close()

            self.plot_one_handover_velocity(trajectory_idx, world_frame = False)
            if save_plots:
                plt.savefig( save_location + 'shared_frame/' + 'plan_share_frame_vel' + str(trajectory_idx) + '.png' )
                plt.close()

            self.plot_one_handover_acceleration(trajectory_idx, world_frame = False)
            if save_plots:
                plt.savefig( save_location + 'shared_frame/' + 'plan_share_frame_acc' + str(trajectory_idx) + '.png' )
                plt.close()

        #self.plot_legend_pos()
        #if save_plots:
        #    #plt.show(block=True)
        #    plt.savefig( save_location + 'world_frame/' + 'legend_pos' + str(trajectory_idx) + '.png' )
        #    plt.close()
        
        #self.plot_title_pos()
        #if save_plots:
        #    #plt.show(block=True)
        #    plt.savefig( save_location + 'world_frame/' + 'title_pos' + str(trajectory_idx) + '.png' )
        #    plt.close()

        if show_plots:
            plt.show()

    def plot_legend_pos(self):
        """
            Plot a separate legend for Position plots
            NOT IMPLEMENTED
        """
        pass

    def plot_title_pos(self):
        """
            Plot a separate title for Position plots
            NOT IMPLEMENTED
        """
        pass

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

    def plot_x_position(self, trajectory_idx, world_frame = True):
        """
            Plot the x positipn only of a handover
        """
        current_step = trajectory_idx
        
        chest_Taker = np.array( [self.x_taker_chest_list, self.y_taker_chest_list, self.z_taker_chest_list] )
        chest_Giver = np.array( [self.x_giver_chest_list, self.y_giver_chest_list, self.z_giver_chest_list] )

        xg = self.x_giver_in_mid[trajectory_idx, :]
        yg = self.y_giver_in_mid[trajectory_idx, :]
        zg = self.z_giver_in_mid[trajectory_idx, :]
        xt = self.x_taker_in_mid[trajectory_idx, :]
        yt = self.y_taker_in_mid[trajectory_idx, :]
        zt = self.z_taker_in_mid[trajectory_idx, :]

        xg_world = np.zeros(xg.shape)
        xt_world = np.zeros(xt.shape)

        final_xg = self.x_true_giver_list
        final_yg = self.y_true_giver_list
        final_zg = self.z_true_giver_list
        final_xt = self.x_true_taker_list
        final_yt = self.y_true_taker_list
        final_zt = self.z_true_taker_list

        final_xg_world = np.zeros(xg.shape)
        final_xt_world = np.zeros(xt.shape)

        handover_point = self.predicted_handover_n_list[trajectory_idx]
        handover_distance = self.predicted_handover_K_list[trajectory_idx]

        if world_frame:
            # Transform
            for t in range(self.N):
                if t < self.no_trajectories:
                    TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
        
                P_n = np.matmul(TF, np.array([xg[t], yg[t], zg[t], 1]))  #giver point at time n
                xg_world[t] = P_n[0]

                P_n = np.matmul(TF, np.array([xt[t], yt[t], zt[t], 1]))  #taker point at time n
                xt_world[t] = P_n[0]

                #if t < self.M:
                P_n = np.matmul(TF, np.array([final_xg[t], final_yg[t], final_zg[t], 1]))  #final trajectory of handover, human
                final_xg_world[t] = P_n[0]

                P_n = np.matmul(TF, np.array([final_xt[t], final_yt[t], final_zt[t], 1]))  #final trajectory of handover, robot
                final_xt_world[t] = P_n[0]
        else:
            # Don't transform
            xg_world = xg
            xt_world = xt
            final_xg_world = final_xg
            final_xt_world = final_xt

        #fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zr_plot'     ],
        #                                       ['x',        'y',        'zr_plot'     ],
        #                                       ['x',        'y',        'zr_plot'     ],
        #                                       ['x',        'y',        'zh_plot'     ],
        #                                       ['x',        'y',        'zh_plot'     ],
        #                                       ['x',        'y',        'zh_plot'     ],
        #                                       ['legend',   'legend',   'legend'],
        #                                       ['legend',   'legend',   'legend'],
        #                                       ['legend',   'legend',   'legend']], layout='tight')

        fig_pos = plt.figure()
        ax_pos_x = plt.axes()

        #ax_pos_x = ax_dict['x']
        #ax_pos_y = ax_dict['y']
        #ax_pos_zh = ax_dict['zh_plot']
        #ax_pos_zr = ax_dict['zr_plot']
        ax_pos_x.grid(True)
        #ax_pos_y.grid(True)
        #ax_pos_zh.grid(True)
        #ax_pos_zr.grid(True)

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

        # Title
        if world_frame:
            ax_pos_x.set_title('Handover trajectory in world frame, n = ' + str(current_step))
        else:
            ax_pos_x.set_title('Handover trajectory in shared frame, n = ' + str(current_step))

        # Legend
        #ax_dict['legend'].set_visible(False)
        #ax_dict['legend'].set_box_aspect(0.001)
        #handles, labels = ax_pos_y.get_legend_handles_labels()
        # Format for screen
        #fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=4)
        
        # Format for ikra paper
        #fig_pos.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05),shadow=True, ncol=2)

        # Axis limits
        #ax_pos_x.set_ylim(0.0, 1.75)
        #ax_pos_y.set_ylim(-0.7, 0.7)
        #ax_pos_zh.set_ylim(-0.3, 0.4)
        #ax_pos_zr.set_ylim(-0.3, 0.4)

        ax_pos_x.set_xlim(-0.2, (self.N+1) * self.dt)
        #ax_pos_y.set_xlim(-0.2, (self.N+1) * self.dt)
        #ax_pos_zh.set_xlim(-0.2, (self.N+1) * self.dt)
        #ax_pos_zr.set_xlim(-0.2, (self.N+1) * self.dt)

        # Window size (important for consistent figures)
        # Format for screen
        #fig_pos.set_figheight(6)
        #fig_pos.set_figwidth(12)
        
        # Format for ikra paper
        fig_pos.set_figheight(9)
        fig_pos.set_figwidth(10)



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

        # # # # plot the figure # # # #
        # make figure
        # Format for screen
        #fig_pos, ax_dict = plt.subplot_mosaic([['x',        'y',        'zr_plot'     ],
        #                                       ['x',        'y',        'zr_plot'     ],
        #                                       ['x',        'y',        'zr_plot'     ],
        #                                       ['x',        'y',        'zh_plot'     ],
        #                                       ['x',        'y',        'zh_plot'     ],
        #                                       ['x',        'y',        'zh_plot'     ],
        #                                       ['legend',   'legend',   'legend'],
        #                                       ['legend',   'legend',   'legend']], layout='tight')

        # Format for ikra paper
        #fig_pos, ax_dict = plt.subplot_mosaic([['x', 'x'],                
        #                                       ['x', 'x'],                
        #                                       ['x', 'x'],                                                                
        #                                       ['y', 'y'],
        #                                       ['y', 'y'],
        #                                       ['y', 'y'],
        #                                       ['zr_plot', 'zh_plot'],
        #                                       ['zr_plot', 'zh_plot'],
        #                                       ['zr_plot', 'zh_plot'],
        #                                       ['legend','legend'],
        #                                       ['legend','legend'],
        #                                       ['legend','legend']], layout='tight')
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
      
        ax_pos_y.plot([t[n] for n in range(0, self.N)], final_yg_world[0:self.N], '--m', alpha=0.5, label='True giver')
        ax_pos_y.plot([t[n] for n in range(0, self.N)], final_yt_world[0:self.N], '--b', alpha=0.5, label='True taker')
    
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
            ax_pos_y.set_title('Handover trajectory in world frame, n = ' + str(current_step))
        else:
            ax_pos_y.set_title('Handover trajectory in shared frame, n = ' + str(current_step))

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


        
            #plt.show()

# Functions for easy access
#def plot_recorded_handovers(dt, robot_role, x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, 
#                                            x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, 
#                                            x_giver_chest_list, y_giver_chest_list, z_giver_chest_list,
#                                            x_taker_chest_list, y_taker_chest_list, z_taker_chest_list,
#                                            predicted_handover_n_list, predicted_handover_K_list):
#    """
#        Show plot of all handovers
#    """
#    p = plotRecordedHandover(dt, robot_role)
#    p.set_variables_from_lists(x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, x_giver_chest_list, y_giver_chest_list, z_giver_chest_list, x_taker_chest_list, y_taker_chest_list, z_taker_chest_list, predicted_handover_n_list, predicted_handover_K_list)
#    p.plot_all_handovers(show_plots = True, save_plots = False)

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
    """
        Save plots of all handovers
    """
    p = plotRecordedHandover(dt, robot_role)
    p.set_variables_from_lists(x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, x_giver_chest_list, y_giver_chest_list, z_giver_chest_list, x_taker_chest_list, y_taker_chest_list, z_taker_chest_list, predicted_handover_n_list, predicted_handover_K_list, trajectory_list, dimensions,x_true_giver_list, y_true_giver_list, z_true_giver_list, x_true_taker_list, y_true_taker_list, z_true_taker_list)
    p.plot_all_handovers(show_plots = False, save_plots = True)

## Functions for easy access
#def plot_recorded_csv_handovers(dt, robot_role, filepath = ''):
#    """
#        Show plot of all handovers saved in csv's in filepath
#        set robot_role as 'Giver' or 'Taker'
#    """
#    p = plotRecordedHandover(dt, robot_role)
#    p.set_variables_from_csv(filepath)
#    p.plot_all_handovers(show_plots = True, save_plots = False)

#def save_plots_recorded_csv_handovers(dt, robot_role, filepath = '', save_path = ''):
#    """
#        Save plots of all handovers saved in csv's in filepath.
#        Save to png in save_path
#        set robot_role as 'Giver' or 'Taker'
#    """
#    p = plotRecordedHandover(dt, robot_role)
#    p.set_variables_from_csv(filepath)
#    p.plot_all_handovers(show_plots = False, save_plots = True, save_location = save_path )

#def plot_recorded_handovers(dt, robot_role, x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, 
#                                            x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, 
#                                            x_giver_chest_list, y_giver_chest_list, z_giver_chest_list,
#                                            x_taker_chest_list, y_taker_chest_list, z_taker_chest_list,
#                                            predicted_handover_n_list, predicted_handover_K_list):
#    """
#        Show plot of all handovers
#        set robot_role as 'Giver' or 'Taker'
#    """
#    p = plotRecordedHandover(dt, robot_role)
#    p.set_variables_from_lists(x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, x_giver_chest_list, y_giver_chest_list, z_giver_chest_list, x_taker_chest_list, y_taker_chest_list, z_taker_chest_list, predicted_handover_n_list, predicted_handover_K_list)
#    p.plot_all_handovers(show_plots = True, save_plots = False)

#def save_plot_recorded_handovers(dt, robot_role, x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, 
#                                                 x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, 
#                                                 x_giver_chest_list, y_giver_chest_list, z_giver_chest_list,
#                                                 x_taker_chest_list, y_taker_chest_list, z_taker_chest_list):
#    """
#        Show plot of all handovers
#        set robot_role as 'Giver' or 'Taker'
#    """
#    p = plotRecordedHandover(dt, robot_role)
#    p.set_variables_from_lists(x_giver_in_mid, y_giver_in_mid, z_giver_in_mid, x_taker_in_mid, y_taker_in_mid, z_taker_in_mid, x_giver_chest_list, y_giver_chest_list, z_giver_chest_list, x_taker_chest_list, y_taker_chest_list, z_taker_chest_list)
#    p.plot_all_handovers(show_plots = False, save_plots = True)


 # Demo
#if __name__ == '__main__':
#    # Load recorded trajectory data
#    filepath = 'G:/My Drive/Exjobb Drive folder/Exjobb/20230814-Handover_data/20230817/'
    
#    #filepath = ''

#    savelocation = filepath

#    robot_role = 'Taker'

#    dt=0.2

#    # Run plot function
#    #plot_recorded_handovers(dt, robot_role, filepath)

#    save_plots_recorded_csv_handovers(dt, robot_role, filepath, savelocation)