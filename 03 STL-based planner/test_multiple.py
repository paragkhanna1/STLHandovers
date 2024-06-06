#!/usr/bin/env python

import debug_functions

from STLGenerateJointHandover import generate_signal_problem, plot_trajectory, plot_trajectory2, plot_in_worldframe
#from tfplot import plot_in_worldframe
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
from plotRecordedHandover import *
#from testHumanRobotROSbag1 import save_plot_recorded_handovers, plotRecordedHandover
import csv

######## SETTINGS ########
result_directory_name = 'Result_directory'
# Role of giver and taker
#ROBOT_ROLE = 'Taker'
ROBOT_ROLE = 'Giver'

# Update robot position using planer output (True), or recorded data (False)
UPDATE_ROBOT_FROM_PLAN = True

# Recorded handover data
#   Path should lead to handovers in the format of Dataset
#       P. Khanna, M. Bj ̈orkman, and C. Smith
#       “A multimodal data set of human handovers with design implications for human-robot handovers” 
#       2023.
# I use reversed handovers from the dataset as these were not used in the learning process
Validation_dataset_paths = [
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair1/Setting1/Set1/handovers_reverse_P1_S1', 
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair1/Setting1/Set2/handovers_reverse_P1_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair2/Setting1/Set1/handovers_reverse_P2_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair2/Setting1/Set2/handovers_reverse_P2_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair3/Setting1/Set1/handovers_reverse_P3_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair3/Setting1/Set2/handovers_reverse_P3_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair4/Setting1/Set2/handovers_reverse_P4_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair5/Setting1/Set2/handovers_reverse_P5_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair6/Setting1/Set1/handovers_reverse_P6_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair6/Setting1/Set2/handovers_reverse_P6_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair7/Setting1/Set1/handovers_reverse_P7_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair7/Setting1/Set2/handovers_reverse_P7_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair8/Setting1/Set1/handovers_reverse_P8_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair8/Setting1/Set2/handovers_reverse_P8_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair9/Setting1/Set1/handovers_reverse_P9_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair9/Setting1/Set2/handovers_reverse_P9_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair10/Setting1/Set1/handovers_reverse_P10_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair10/Setting1/Set2/handovers_reverse_P10_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair11/Setting1/Set1/handovers_reverse_P11_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair11/Setting1/Set2/handovers_reverse_P11_S2',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair12/Setting1/Set1/handovers_reverse_P12_S1',
                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair12/Setting1/Set2/handovers_reverse_P12_S2',
                            ]

# Paths for heavy objects (Setting 2)
#Validation_dataset_paths = ['D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair1/Setting2/Set3/handovers_P1_S3',
#                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair2/Setting2/Set3/handovers_P2_S3',
#                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair3/Setting2/Set3/handovers_P3_S3',
#                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair4/Setting2/Set3/handovers_P4_S3']
## Path for blind handovers (Setting 3)
#Validation_dataset_paths = ['D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair1/Setting3/Set5/handovers_P1_S5',
#                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair2/Setting3/Set5/handovers_P2_S5',
#                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair3/Setting3/Set5/handovers_P3_S5',
#                            'D:/Electrical Engineering/Offline Exjobb/Handover dataset/dataset-main/dataset_pairwise/Pair4/Setting3/Set5/handovers_P4_S5']


#RECORDED_DATA_PATH = 'Validation_data/handovers_P1_S1/handover_2/'  # From learning process validation data
#RECORDED_DATA_PATH = 'Validation_data/handovers_P2_S1/handover_27/' # From learning process validation data
#RECORDED_DATA_PATH = 'Validation_data/handovers_P4_S3/handover_10/' # From heavy object set, never used in learning!

# Manual set of handover start time
#   Set to a time around where the giver starts moving. -1.2 is a resonable value
handover_start_time = -1.2

# Choose STL specifications
STL100 = True      # Each predicate satisfies all validation data
STL95_100 = True   # Each predicate satisfies [95, 100) % of validation data
STL90_95 = True    # Each predicate satisfies [90, 95) % of validation data

STLDebug = False

KSHIRSAGAR_APPROACH = True        # Use Kshirsagar inspired approach stratergy    F[0,t] (K_p < epsilon)
Kshirsagar_only = False           # Ignore STL_list, only use Kshirsagar approach
HARD_KSHIRSAGAR_CONSTRAINT = True # Robustness (pseudo-robustness) of Kshirsagar part must be positive
HARD_STL_CONSTRAINT= False        # Robustness (pseudo-robustness) of all STL must be positive at every step 


dt = 0.2 # [s] step time

######## END OF SETTINGS ########

# Log handover statistics to csv
result_header = ['Handover', 
                 'Location_x_1st', 'Location_y_1st', 'Location_z_1st', 
                 'Location_error_1st', 
                 'Time_1st', 'Time_error_1st', 
                 'Max_vel_1st', 'Max_acc_1st',
                 'RMSE_trajectory_1st',
                 'Success_final', 
                 'Location_x_final', 'Location_y_final', 'Location_z_final', 
                 'Location_error_final', 
                 'Time_final', 'Time_error_final',
                 'Max_vel_final', 'Max_acc_final',
                 'RMSE_trajectory_final'
                 ]
#location_error_over_time_W
#location_error_over_time_O
#time_error_over_time
#max_vel_over_time
#max_acc_over_time
#RMSE_over_time_O
#RMSE_over_time_W
# Log handover location error over time (Shared frame)
result_location_error_header_O = ['Handover', 'Successful',
                                'Location error in shared frame planned at t_0', 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8']
# Log handover location error over time (World frame)
result_location_error_header_W = ['Handover', 'Successful',
                                'Location error in world frame planned at t_0', 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8']
# Log handover time error over time
result_time_error_header = ['Handover', 'Successful',
                            'Time error planned at t_0', 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8']
result_max_vel_header =  ['Handover', 'Successful',
                            'Max vel planned at t_0', 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8']
result_max_acc_header =  ['Handover', 'Successful',
                            'Max acc planned at t_0', 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8']
result_RMSE_header_O = ['Handover', 'Successful',
                           'Root Mean Square error in shared frame planned at t_0', 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8']
result_RMSE_header_W = ['Handover', 'Successful',
                           'Root Mean Square error in world frame planned at t_0', 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 't_7', 't_8']

extension = '.csv'
unique_number = 1
while os.path.exists(result_directory_name + '_' + str(unique_number)):
    unique_number = unique_number + 1
dir_name = result_directory_name + '_' + str(unique_number)
os.mkdir(dir_name)

result_location_filename_O = dir_name + '/' + 'Location_error_over_time_O_' + str(unique_number) + extension
with open(result_location_filename_O, 'w') as file:
    result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
    result_writer.writerow(result_location_error_header_O)
result_location_filename_W = dir_name + '/' + 'Location_error_over_time_W_' + str(unique_number) + extension
with open(result_location_filename_W, 'w') as file:
    result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
    result_writer.writerow(result_location_error_header_W)

result_time_filename = dir_name + '/' + 'Time_error_over_time_' + str(unique_number) + extension
with open(result_time_filename, 'w') as file:
    result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
    result_writer.writerow(result_time_error_header)

result_max_vel_filename = dir_name + '/' + 'Max_vel_over_time_' + str(unique_number) + extension
with open(result_max_vel_filename , 'w') as file:
    result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
    result_writer.writerow(result_max_vel_header)
result_max_acc_filename = dir_name + '/' + 'Max_acc_over_time_' + str(unique_number) + extension
with open(result_max_acc_filename , 'w') as file:
    result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
    result_writer.writerow(result_max_acc_header)

result_RMSE_filename_O = dir_name + '/' + 'RMSE_over_time_O_' + str(unique_number) + extension
with open(result_RMSE_filename_O , 'w') as file:
    result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
    result_writer.writerow(result_RMSE_header_O)
result_RMSE_filename_W = dir_name + '/' + 'RMSE_over_time_W_' + str(unique_number) + extension
with open(result_RMSE_filename_W , 'w') as file:
    result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
    result_writer.writerow(result_RMSE_header_W)

result_filename = dir_name + '/' + 'First_Last_plan_results_' + str(unique_number) + extension
with open(result_filename, 'w') as file:
    result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
    result_writer.writerow(result_header)


# START LOOP, Run for all handovers
for Set_path in Validation_dataset_paths:
    for Handover_data_path in os.listdir(Set_path):
        RECORDED_DATA_PATH = Set_path + '/' + Handover_data_path + '/'

        print(RECORDED_DATA_PATH)

        print("dt = "+str(dt))

        

        SHOW = False    #   Show plots
        SAVE = False    #   Save plots at save location

        mpl.rcParams['text.usetex'] = True

        OPTIMIZE_ROBUSTNESS=True
        QUANTITATIVE_OPTIMIZATION= True
        #HARD_STL_CONSTRAINT= True

        # Output path for plots (if SAVE = True)
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        SAVE_PATH = ''
        #PLOT_PATH = 'C:/Users/jfred/FlaoGDrive/Exjobb Drive folder/Exjobb/STL Generate Trajectory MILP/active-learn-stl-master/active-learn-stl-master/Plot_animation_2'
        #PLOT_NAME = f'/RobotHumanROSbag_timestr_HumanGiver_RobotTaker_OptRho{OPTIMIZE_ROBUSTNESS:01}_QuOpt{QUANTITATIVE_OPTIMIZATION:01}_HardSTL{HARD_STL_CONSTRAINT:01}'


        # RECORDED_DATA_PATH = 'C:/Users/jfred/FlaoGDrive/Exjobb Drive folder/Exjobb/ ...
        #RECORDED_DATA = pd.read_csv( RECORDED_DATA_PATH + "P1_S1_validation_handover_2.csv" )

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
        # To get index of any dimension, use dimensions.index('giver_x')
        pick_up_epsilon = 0.1 # proximity to pickup point

        # Find idx for latest timestamp at sample frequency
        sample_idx = lambda time_array : [idx for idx, val in enumerate( 
            [floor(time_array[i+1]/dt) - floor(time_array[i]/dt) for i in range(len(time_array)-1) ] ) if val != 0]

        # LOAD CHEST TRACKING
        #if ROBOT_ROLE == 'Taker':
        #    human_chest_data = pd.read_csv( RECORDED_DATA_PATH + 'giver_chest_pose_saved.csv' )
        #    robot_chest_data = pd.read_csv( RECORDED_DATA_PATH + 'taker_chest_pose_saved.csv' )
        #elif ROBOT_ROLE == 'Giver':
        #    robot_chest_data = pd.read_csv( RECORDED_DATA_PATH + 'giver_chest_pose_saved.csv' )
        #    human_chest_data = pd.read_csv( RECORDED_DATA_PATH + 'taker_chest_pose_saved.csv' )
        giver_chest_data = pd.read_csv( RECORDED_DATA_PATH + 'giver_chest_pose_saved.csv' )
        taker_chest_data = pd.read_csv( RECORDED_DATA_PATH + 'taker_chest_pose_saved.csv' )
        #human_chest_data = np.array( [human_chest_data.x, human_chest_data.y, human_chest_data.z] )
        giver_chest_time_array = np.linspace(-400/120, 400/120, 801)
        #giver_chest_time_array = (giver_chest_time_array - giver_chest_time_array[0])/1e9 # Convert to [s] relative to t_0
        giver_chest_sample_idx = sample_idx( giver_chest_time_array )
        giver_chest_sample_idx.insert(0, 0)

        sample_time = giver_chest_time_array[giver_chest_sample_idx]

        #robot_chest = [0.152, 0, 0.359] # Defines a stationary point in map frame as the chest of the robot

        #taker_chest_data = np.array( [taker_chest_data.x, taker_chest_data.y, taker_chest_data.z] )
        taker_chest_time_array = np.linspace(-400/120, 400/120, 801)
        #giver_chest_time_array = (giver_chest_time_array - giver_chest_time_array[0])/1e9 # Convert to [s] relative to t_0
        taker_chest_sample_idx = sample_idx( taker_chest_time_array )
        taker_chest_sample_idx.insert(0, 0)


        chest_Giver = np.array( [giver_chest_data.x[giver_chest_sample_idx], giver_chest_data.y[giver_chest_sample_idx], giver_chest_data.z[giver_chest_sample_idx]]) # Human chest
        chest_Taker = np.array( [taker_chest_data.x[taker_chest_sample_idx], taker_chest_data.y[taker_chest_sample_idx], taker_chest_data.z[taker_chest_sample_idx]] )


        # Load hand tracking

        # Human hand
        giver_data = pd.read_csv( RECORDED_DATA_PATH + 'giver_RHand_pose_saved.csv' )
        giver_time_array = np.linspace(-400/120, 400/120, 801)
        giver_sample_idx = sample_idx( giver_time_array )
        giver_sample_idx.insert(0, 0)

        # Robot hand
        taker_data = pd.read_csv( RECORDED_DATA_PATH + 'taker_RHand_pose_saved.csv' )
        taker_time_array = np.linspace(-400/120, 400/120, 801)
        taker_sample_idx = sample_idx( taker_time_array )
        taker_sample_idx.insert(0, 0)

        # Get handover location
        True_giver_handover_location_W = np.array( [giver_data.x[400], giver_data.y[400], giver_data.z[400]] )
        True_taker_handover_location_W = np.array( [taker_data.x[400], taker_data.y[400], taker_data.z[400]] )
        Chest_G_at_handover =  np.array( [giver_chest_data.x[400], giver_chest_data.y[400], giver_chest_data.z[400]]) 
        Chest_T_at_handover =  np.array( [taker_chest_data.x[400], taker_chest_data.y[400], taker_chest_data.z[400]]) 
        TF = points2tfmatrix(Chest_G_at_handover , Chest_T_at_handover ) # Transformation matrix at handover time
        True_giver_handover_location_O = np.matmul(TF, np.concatenate( (True_giver_handover_location_W, [1]) ) ) 
        True_taker_handover_location_O = np.matmul(TF, np.concatenate( (True_taker_handover_location_W, [1]) ) ) 
        True_giver_handover_location_O = True_giver_handover_location_O[0:3]
        True_taker_handover_location_O = True_taker_handover_location_O[0:3]
        if ROBOT_ROLE == 'Taker':
            True_handover_location_O = True_taker_handover_location_O
        if ROBOT_ROLE == 'Giver':
            True_handover_location_O = True_giver_handover_location_O
        Handover_time = 400/120


        if ROBOT_ROLE == 'Taker':
            # Downsample to dt
            x_h_n_W = [giver_data.x[i] for i in giver_sample_idx] # Still in map frame
            y_h_n_W = [giver_data.y[i] for i in giver_sample_idx]
            z_h_n_W = [giver_data.z[i] for i in giver_sample_idx]
            # Downsample to dt
            x_r_n_W = [taker_data.x[i] for i in taker_sample_idx] # Still in map frame
            y_r_n_W = [taker_data.y[i] for i in taker_sample_idx]
            z_r_n_W = [taker_data.z[i] for i in taker_sample_idx]
        elif ROBOT_ROLE == 'Giver':
            # Downsample to dt
            x_r_n_W = [giver_data.x[i] for i in giver_sample_idx] # Still in map frame
            y_r_n_W = [giver_data.y[i] for i in giver_sample_idx]
            z_r_n_W = [giver_data.z[i] for i in giver_sample_idx]
            # Downsample to dt
            x_h_n_W = [taker_data.x[i] for i in taker_sample_idx] # Still in map frame
            y_h_n_W = [taker_data.y[i] for i in taker_sample_idx]
            z_h_n_W = [taker_data.z[i] for i in taker_sample_idx]

        # TF the data!!!
        #init variables
        x_h_n_O = x_h_n_W
        y_h_n_O = y_h_n_W
        z_h_n_O = z_h_n_W
        x_r_n_O = x_r_n_W
        y_r_n_O = y_r_n_W
        z_r_n_O = z_r_n_W
        for i in range( chest_Giver.shape[1] ):
            TF = points2tfmatrix(chest_Giver[:,i], chest_Taker[:,i]) # Transformation matrix at time i
    
            if i < len(x_h_n_W):
                P_n = np.matmul(TF, np.array([x_h_n_W[i], y_h_n_W[i], z_h_n_W[i], 1]))  #human point at time n
                x_h_n_O[i] = P_n[0]
                y_h_n_O[i] = P_n[1]
                z_h_n_O[i] = P_n[2] # TF to shared frame

            if i < len(x_r_n_W):
                P_n = np.matmul(TF, np.array([x_r_n_W[i], y_r_n_W[i], z_r_n_W[i], 1]))  #robot point at time n
                x_r_n_O[i] = P_n[0]
                y_r_n_O[i] = P_n[1]
                z_r_n_O[i] = P_n[2] # TF to shared frame

        # "Sim" time
        time_array = [i*dt for i in range(len(x_h_n_O))]

        ########### BUILD STL-model ############
        STL_list = []

        if STLDebug:
            STL_list = STL_list + [
                ["G", 0.20, 0.40, -0.1, 'taker_x', -0.0],
                #["G", 3.20, 4.20, -0.0, 'giver_x', 0.2]
                ["F", 0.6, 1.0, -1.0, 'giver_y', 0.1],
                ]

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
        KSHIRSAGAR_DELTA = [0.3, 0.1, 0.0]# x,y,z offset of giver and taker hands on the object
        KSHIRSAGAR_EPSILON = 0.01
        #KSHIRSAGAR_TIME = 2.0 # [s]
        KSHIRSAGAR_TIME = max_time_in_spec # Get time from max time in learnt STL specification

        #   STL implementation
        t_K = int(KSHIRSAGAR_TIME / dt) + 1
        Spatial_predicate = STLFormula.Predicate('K_p', operatorclass.lt, KSHIRSAGAR_EPSILON, dimensions.index('K_p')) # Fix variable pi_index_signal = 9 if number of signals change
        Kshirsagar_phi = STLFormula.Eventually( Spatial_predicate, 0, t_K)
        if KSHIRSAGAR_APPROACH == True:
            phi = STLFormula.Conjunction(phi, Kshirsagar_phi)

        if Kshirsagar_only:
            phi = Kshirsagar_phi

        # Remove negations
        phi_nnf = STLFormula.toNegationNormalForm(phi, False)


        # Planner parameters
        domain=[-10, 10] # Domain of signals
        max_speed = 1 # m/s
        max_giver_acceleration = [6.8506, 3.8406, 8.3751] # m/s^2
        max_taker_acceleration = [6.1479, 4.2769, 7.7459] # m/s^2
        #max_giver_acceleration = [10, 10, 10] # m/s^2
        #max_taker_acceleration = [10, 10, 10] # m/s^2
        max_human_speed = 1 # m/s
        max_change = [0,0,0, # speed limits, not used
                        max_giver_acceleration[0] * dt, max_giver_acceleration[1] * dt, max_giver_acceleration[2] * dt, # Giver acceleration limits
                        0,0,0, # speed limits, not used
                        max_taker_acceleration[0] * dt, max_taker_acceleration[1] * dt, max_taker_acceleration[2] * dt] # Taker acceleration limits
        U = [max_giver_acceleration[0], max_giver_acceleration[1], max_giver_acceleration[2],
             max_taker_acceleration[0], max_taker_acceleration[1], max_taker_acceleration[2]]


        # Starting parameters
        if ROBOT_ROLE == 'Taker':
            start=[x_h_n_O[0], y_h_n_O[0], z_h_n_O[0], # Giver position
                x_r_n_O[0], y_r_n_O[0], z_r_n_O[0], # Taker position
                ]
        else:
            start=[x_r_n_O[0], y_r_n_O[0], z_r_n_O[0], # Giver position
                x_h_n_O[0], y_h_n_O[0], z_h_n_O[0], # Taker position
                ]

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

        location_error_over_time_W  = [None] * t_K
        location_error_over_time_O  = [None] * t_K
        time_error_over_time    = [None] * t_K
        max_vel_over_time   = [None] * t_K
        max_acc_over_time   = [None] * t_K
        RMSE_over_time_O    = [None] * t_K
        RMSE_over_time_W    = [None] * t_K

        # State machine
        #   0: Idle
        #   4: in pickup position
        #   1: Reach phase
        #   2: Handover failed (Reset)
        #   3: Handover success (Reset)
        state_machine = 0

        # Execution loop
        n = 0

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

        # Pick up position
        # The position of the object to pick up
        i = int(0) # picked from the graph!
        start_x = x_h_n_O[i]
        start_y = y_h_n_O[i]
        start_z = z_h_n_O[i]





        # Save predictions and print later
        human_chest_list=[]
        robot_chest_list=[]
        x_human_chest_list = []
        y_human_chest_list = []
        z_human_chest_list = []
        x_robot_chest_list = []
        y_robot_chest_list = []
        z_robot_chest_list = []
        x_robot_in_mid = []
        y_robot_in_mid = []
        z_robot_in_mid = []
        x_human_in_mid = []
        y_human_in_mid = []
        z_human_in_mid = []
        trajectory_list = []

        x_true_human_list=[]
        y_true_human_list=[]
        z_true_human_list=[]
        x_true_robot_list=[]
        y_true_robot_list=[]
        z_true_robot_list=[]

        #n_list = []
        predicted_handover_n_list = []
        predicted_handover_K_list = []

        #in_pickup_zone_condition = (t == int((6.2)/dt)-1 or t==int(18/dt)-1 or t==int(27.6/dt)-1 or t==int(38.6/dt)-1 or t==int(45.6/dt)-1 ) 
        #starting_condition = (t == int((6.2)/dt) or t==int(18/dt) or t==int(27.6/dt) or t==int(38.6/dt) or t==int(45.6/dt)) 
        success_condition = 0.1 # \delta_success, K_t_n distance for successfull handover
        for t in range(0, len( x_h_n_O ) ): #len( x_h_n_O )): # Loop through starting points

            if state_machine == 0: # Idle
                # Set variables
                t_0 = t # Start of handover set to current time
                n = 0 # Index of the current step of execution

                ## Set/fix current human position values (Recorded)
                #problem.unfix_variables([n], human_pos_idx)
                #problem.fix_variables([n], human_pos_idx, [[x_h_n_O[t], y_h_n_O[t], z_h_n_O[t]]])
                ## Set/fix robot current position (Recorded)
                #problem.unfix_variables([n], robot_pos_idx)
                #problem.fix_variables([n], robot_pos_idx, [[x_r_n_O[t], y_r_n_O[t], z_r_n_O[t]]])

                ### NEW
                # update past states s[-2]=s[-1]
                # update past states s[-1]=s[0]
                # set current state  s[0]=current position
        
                if ROBOT_ROLE == 'Taker':
                    current_giver_shared = [x_h_n_O[t], y_h_n_O[t], z_h_n_O[t]]
                    current_taker_shared = [x_r_n_O[t], y_r_n_O[t], z_r_n_O[t]]
                elif ROBOT_ROLE == 'Giver':
                    current_giver_shared = [x_r_n_O[t], y_r_n_O[t], z_r_n_O[t]]
                    current_taker_shared = [x_h_n_O[t], y_h_n_O[t], z_h_n_O[t]]

                problem.set_leading_states(current_giver_shared, current_taker_shared)


                # Execute planner at t_n = 0
                start_execution_time = time.monotonic()
                solution_found = problem.generate_path(n)
                execution_time = time.monotonic() - start_execution_time
                trajectory = problem.get_path()
                debug_functions.check_state_evolution_encoding(problem.s, dimensions, dt)

                # Move robot to "Predicted start position"
                #problem.fix_variables([n], robot_pos_idx, [[trajectory[0][robot_pos_idx[0]], trajectory[0][robot_pos_idx[1]], trajectory[0][robot_pos_idx[2]]]]) 

                # Check starting conditions
                #starting_condition = (x_h_n_O[t] > -0.3)  # To be implemented
                #starting_condition = ( (t == int(7/dt)) or (t == int(11/dt)) or (t == int(15/dt)) or (t == int(20/dt)) or (t == int(25/dt)) ) # (Identified manually at)
                #starting_condition = (t == int(18/dt))
                #starting_condition = t == int(38.6/dt) or t==int(45.6/dt) # t==int(27.4/dt)
                #in_pickup_zone_condition = (t == int((6.2)/dt)-1 or t==int(18/dt)-1 or t==int(27.6/dt)-1 or t==int(38.6/dt)-1 or t==int(45.6/dt)-1 ) 
                in_pickup_zone_condition = abs(sample_time[t] - (handover_start_time - 0.5)) < dt/2
                #starting_condition = (t == int((6.2)/dt) or t==int(18/dt) or t==int(27.6/dt) or t==int(38.6/dt) or t==int(45.6/dt)) 
                if in_pickup_zone_condition:
                    state_machine = 4
    
            elif state_machine == 4: # in pickup position
                print("In pickup zone")
                # Set variables
                t_0 = t # Start of handover set to current time
                n = 0 # Index of the current step of execution

                ### NEW
                ## Set/fix current human position values (Recorded)
                #problem.unfix_variables([n], human_pos_idx)
                #problem.fix_variables([n], human_pos_idx, [[x_h_n_O[t], y_h_n_O[t], z_h_n_O[t]]])
                ## Set/fix robot current position (Recorded)
                #problem.unfix_variables([n], robot_pos_idx)
                #problem.fix_variables([n], robot_pos_idx, [[x_r_n_O[t], y_r_n_O[t], z_r_n_O[t]]])
                if ROBOT_ROLE == 'Taker':
                    current_giver_shared = [x_h_n_O[t], y_h_n_O[t], z_h_n_O[t]]
                    current_taker_shared = [x_r_n_O[t], y_r_n_O[t], z_r_n_O[t]]
                elif ROBOT_ROLE == 'Giver':
                    current_giver_shared = [x_r_n_O[t], y_r_n_O[t], z_r_n_O[t]]
                    current_taker_shared = [x_h_n_O[t], y_h_n_O[t], z_h_n_O[t]]
                problem.set_leading_states(current_giver_shared, current_taker_shared)

                # Execute planner at t_n = 0
                start_execution_time = time.monotonic()
                solution_found = problem.generate_path(n)
                execution_time = time.monotonic() - start_execution_time
                trajectory = problem.get_path()
        
                debug_functions.check_state_evolution_encoding(problem.s, dimensions, dt)

                # Check starting conditions
                #starting_condition = (x_h_n_O[t] > -0.3)  # To be implemented
                #starting_condition = ( (t == int(7/dt)) or (t == int(11/dt)) or (t == int(15/dt)) or (t == int(20/dt)) or (t == int(25/dt)) ) # (Identified manually at)
                #starting_condition = (t == int(6/dt) or t==int(17/dt) or t==int(27/dt) or t==int(38/dt)) 
                #starting_condition = (t == int(18/dt))
                #starting_condition = t == int(38.6/dt) or t==int(45.6/dt) # t==int(27.4/dt)
        
                # eucl distance from pickup start position
                #starting_condition = pick_up_epsilon > sqrt((x_h_n_O[t]-start_x)**2 + (y_h_n_O[t]-start_y)**2 + (z_h_n_O[t]-start_z)**2)
                #in_pickup_zone_condition = (t == int((6.2)/dt)-1 or t==int(18/dt)-1 or t==int(27.6/dt)-1 or t==int(38.6/dt)-1 or t==int(45.6/dt)-1 ) 
                #starting_condition = (t == int((6.2)/dt) or t==int(18/dt) or t==int(27.6/dt) or t==int(38.6/dt) or t==int(45.6/dt)) 
                starting_condition = abs(sample_time[t] - (handover_start_time)) < dt/2
                if starting_condition:
                    state_machine = 1

                    ### NEW
                    ### TODO use outputs of plan
                    # Get controll output
                    if ROBOT_ROLE == 'Taker':
                        robot_next_position = problem.get_taker_plan(n+1)
                    if ROBOT_ROLE == 'Giver':
                        robot_next_position = problem.get_giver_plan(n+1)

                    # Save 1st trajectory
                    if ROBOT_ROLE == 'Taker':
                        x_robot_chest_list.append( chest_Taker[0,t] )
                        y_robot_chest_list.append( chest_Taker[1,t] )
                        z_robot_chest_list.append( chest_Taker[2,t] )
                        x_human_chest_list.append( chest_Giver[0,t] )
                        y_human_chest_list.append( chest_Giver[1,t] )
                        z_human_chest_list.append( chest_Giver[2,t] )
                    elif ROBOT_ROLE == 'Giver':
                        x_human_chest_list.append( chest_Taker[0,t] )
                        y_human_chest_list.append( chest_Taker[1,t] )
                        z_human_chest_list.append( chest_Taker[2,t] )
                        x_robot_chest_list.append( chest_Giver[0,t] )
                        y_robot_chest_list.append( chest_Giver[1,t] )
                        z_robot_chest_list.append( chest_Giver[2,t] )
                    x_robot_in_mid.append( [ trajectory[a][robot_pos_idx[0]] for a in range(0,phi.horizon+1) ] )
                    y_robot_in_mid.append( [ trajectory[a][robot_pos_idx[1]] for a in range(0,phi.horizon+1) ] )
                    z_robot_in_mid.append( [ trajectory[a][robot_pos_idx[2]] for a in range(0,phi.horizon+1) ] )
                    x_human_in_mid.append( [ trajectory[a][human_pos_idx[0]] for a in range(0,phi.horizon+1) ] )
                    y_human_in_mid.append( [ trajectory[a][human_pos_idx[1]] for a in range(0,phi.horizon+1) ] )
                    z_human_in_mid.append( [ trajectory[a][human_pos_idx[2]] for a in range(0,phi.horizon+1) ] )
                    trajectory_list.append( trajectory )
            
                    x_true_human_list.append( x_h_n_O[t] )
                    y_true_human_list.append( y_h_n_O[t] )
                    z_true_human_list.append( z_h_n_O[t] )
                    x_true_robot_list.append( x_r_n_O[t] )
                    y_true_robot_list.append( y_r_n_O[t] )
                    z_true_robot_list.append( z_r_n_O[t] )

                    n_handover = problem.get_predicted_handover_step( 0.1, n ) # Handover predicted at t_n_handover in the 1st plan
                    predicted_handover_n_list.append( n_handover )
                    predicted_handover_K_list.append( problem.get_K(n_handover) )

                    # Get statistics for first plan (open loop)
                    # Handover location (Predicted in shared frame (_O), transformed to world frame (_W) using chest positions at time of planning)
                    Handover_location_1st_plan_O = np.array( [ trajectory[n_handover][robot_pos_idx[0]], trajectory[n_handover][robot_pos_idx[1]], trajectory[n_handover][robot_pos_idx[2]] ] )
                    TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
                    Handover_location_1st_plan_W = np.matmul(TF, np.concatenate( (Handover_location_1st_plan_O , [1]) ))[0:3]
                    # Handover time
                    Handover_time_1st_plan = (n_handover + t_0) * dt 
                    # Handover location accuracy
                    Handover_location_error_1st_plan_W = np.sum( (True_giver_handover_location_W - Handover_location_1st_plan_W)**2 )
                    Handover_location_error_1st_plan_O = np.sum( (True_giver_handover_location_O - Handover_location_1st_plan_O)**2 )
                    # Handover time accuracy
                    Handover_time_error_1st_plan = Handover_time_1st_plan - Handover_time 
                    # RMSE trajectory
                    true_path_O = np.array([x_r_n_O[t_0:t_0+n_handover+1], y_r_n_O[t_0:t_0+n_handover+1], z_r_n_O[t_0:t_0+n_handover+1]]).T
                    true_path_W = np.array([x_r_n_W[t_0:t_0+n_handover+1], y_r_n_W[t_0:t_0+n_handover+1], z_r_n_W[t_0:t_0+n_handover+1]]).T
                    planed_path_O = np.array(trajectory)[0:n_handover+1, robot_pos_idx]
                    planed_path_W = np.zeros(planed_path_O.shape)
                    TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
                    for i in range(planed_path_O.shape[0]):
                        planed_path_W[i,:] = np.matmul(TF, np.concatenate( (planed_path_O[i,:], [1]) ))[0:3]
                    RMSE_trajectory_1st_plan_O = np.sqrt( np.sum((true_path_O - planed_path_O)**2) / (n_handover+1) )
                    RMSE_trajectory_1st_plan_W = np.sqrt( np.sum((true_path_W - planed_path_W)**2) / (n_handover+1) )
                    # Max velocity
                    if ROBOT_ROLE == 'Giver':
                        robot_vel_idx = [dimensions.index('giver_dx'), dimensions.index('giver_dy'), dimensions.index('giver_dz')]
                        robot_acc_idx = [dimensions.index('giver_ddx'), dimensions.index('giver_ddy'), dimensions.index('giver_ddz')]
                    if ROBOT_ROLE == 'Taker':
                        robot_vel_idx = [dimensions.index('taker_dx'), dimensions.index('taker_dy'), dimensions.index('taker_dz')]
                        robot_acc_idx = [dimensions.index('taker_ddx'), dimensions.index('taker_ddy'), dimensions.index('taker_ddz')]
                    Handover_max_vel_1st_plan = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover+1, robot_vel_idx]**2, 1 )))
                    Handover_max_acc_1st_plan = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover+1, robot_acc_idx]**2, 1 )))

                    # Record handover over time performance
                    location_error_over_time_W[n] = Handover_location_error_1st_plan_W
                    location_error_over_time_O[n] = Handover_location_error_1st_plan_O
                    time_error_over_time[n]       = Handover_time_error_1st_plan
                    max_vel_over_time[n] = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover+1, robot_vel_idx]**2, 1 )))
                    max_acc_over_time[n] = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover+1, robot_acc_idx]**2, 1 )))
                    RMSE_over_time_O[n] = RMSE_trajectory_1st_plan_O
                    RMSE_over_time_W[n] = RMSE_trajectory_1st_plan_W

                    if (SAVE or SHOW):
                        plot_trajectory2(phi_nnf, n , trajectory , dt , ROBOT_ROLE , close_flag=False, true_human_x = [x_h_n_O[t_0 + i] for i in range(phi.horizon+1)], true_human_y = [y_h_n_O[t_0 + i] for i in range(phi.horizon+1)], true_human_zG = [z_h_n_O[t_0 + i] for i in range(phi.horizon+1)] )
                    if SAVE:
                        plt.savefig(SAVE_PATH + f'tstart{t_0:}_n{n:}_t{t:}.png' )
                    if SHOW:
                        plt.show()
                    plt.close()
    
            elif state_machine == 1: # Reach phase
                print('Reach phase , SMC=',state_machine)
                # Set variables
                n += 1

                # Set/fix current human position values (Recorded)
                problem.fix_variables([n], human_pos_idx, [[x_h_n_O[t], y_h_n_O[t], z_h_n_O[t]]])
        
        
                # Fix robot to previous control output 
                #   R_{t_n} <-- \hat{R}^{t_{n-1}}_{t_n}
                #   Where \hat{R}^{t_{n-1}}_{t_n} is the robots predicted position at t_n, calculated at t_{n-1}
                #   and R_{t_n} is the true position
                #problem.fix_variables([n], robot_pos_idx)


                if UPDATE_ROBOT_FROM_PLAN:
                    ### Update robot from plan
                    problem.fix_variables([n], robot_pos_idx, [robot_next_position]) ### Plan update
                else:
                    ### Update robot from recorded human
                    problem.fix_variables([n], robot_pos_idx, [[x_r_n_O[t], y_r_n_O[t], z_r_n_O[t]]])


                # Execute planner at t_n = n
                start_execution_time = time.monotonic()
                solution_found = problem.generate_path(n)
                execution_time = time.monotonic() - start_execution_time
                trajectory = problem.get_path()
                debug_functions.check_state_evolution_encoding(problem.s, dimensions, dt)
                
                # Get controll output
                if ROBOT_ROLE == 'Taker':
                    robot_next_position = problem.get_taker_plan(n+1)
                if ROBOT_ROLE == 'Giver':
                    robot_next_position = problem.get_giver_plan(n+1)
    
                # Save trajectories
                if ROBOT_ROLE == 'Taker':
                    x_robot_chest_list.append( chest_Taker[0,t] )
                    y_robot_chest_list.append( chest_Taker[1,t] )
                    z_robot_chest_list.append( chest_Taker[2,t] )
                    x_human_chest_list.append( chest_Giver[0,t] )
                    y_human_chest_list.append( chest_Giver[1,t] )
                    z_human_chest_list.append( chest_Giver[2,t] )        
                elif ROBOT_ROLE == 'Giver':
                    x_human_chest_list.append( chest_Taker[0,t] )
                    y_human_chest_list.append( chest_Taker[1,t] )
                    z_human_chest_list.append( chest_Taker[2,t] )
                    x_robot_chest_list.append( chest_Giver[0,t] )
                    y_robot_chest_list.append( chest_Giver[1,t] )
                    z_robot_chest_list.append( chest_Giver[2,t] )
                x_robot_in_mid.append( [ trajectory[a][robot_pos_idx[0]] for a in range(0,phi.horizon+1) ] )
                y_robot_in_mid.append( [ trajectory[a][robot_pos_idx[1]] for a in range(0,phi.horizon+1) ] )
                z_robot_in_mid.append( [ trajectory[a][robot_pos_idx[2]] for a in range(0,phi.horizon+1) ] )
                x_human_in_mid.append( [ trajectory[a][human_pos_idx[0]] for a in range(0,phi.horizon+1) ] )
                y_human_in_mid.append( [ trajectory[a][human_pos_idx[1]] for a in range(0,phi.horizon+1) ] )
                z_human_in_mid.append( [ trajectory[a][human_pos_idx[2]] for a in range(0,phi.horizon+1) ] )
                trajectory_list.append( trajectory )
        
                x_true_human_list.append( x_h_n_O[t] )
                y_true_human_list.append( y_h_n_O[t] )
                z_true_human_list.append( z_h_n_O[t] )
                x_true_robot_list.append( x_r_n_O[t] )
                y_true_robot_list.append( y_r_n_O[t] )
                z_true_robot_list.append( z_r_n_O[t] )
        
                n_handover = problem.get_predicted_handover_step( 0.1, n )
                predicted_handover_n_list.append( n_handover )
                predicted_handover_K_list.append( problem.get_K(n_handover) )

                # Get performance of plan at this time step
                # Handover location
                Handover_location_n_O = np.array( [ trajectory[n_handover][robot_pos_idx[0]], trajectory[n_handover][robot_pos_idx[1]], trajectory[n_handover][robot_pos_idx[2]] ] )
                TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
                Handover_location_n_W = np.matmul(TF, np.concatenate( (Handover_location_n_O, [1]) ))[0:3]
                # Handover time
                Handover_time_n = (n_handover + t_0) * dt 
                # Handover location accuracy
                Handover_location_error_n_W = np.sum( (True_giver_handover_location_W - Handover_location_n_W)**2 )
                Handover_location_error_n_O = np.sum( (True_giver_handover_location_O - Handover_location_n_O)**2 )
                # Handover time accuracy
                Handover_time_error_n = Handover_time_n - Handover_time 
                # RMSE trajectory (shared )
                true_path_O = np.array([x_r_n_O[t_0:t_0+n_handover+1], y_r_n_O[t_0:t_0+n_handover+1], z_r_n_O[t_0:t_0+n_handover+1]]).T
                true_path_W = np.array([x_r_n_W[t_0:t_0+n_handover+1], y_r_n_W[t_0:t_0+n_handover+1], z_r_n_W[t_0:t_0+n_handover+1]]).T
                planed_path_O = np.array(trajectory)[0:n_handover+1, robot_pos_idx]
                planed_path_W = np.zeros(planed_path_O.shape)
                TF = points2invtfmatrix(chest_Giver[:,t], chest_Taker[:,t])
                for i in range(planed_path_O.shape[0]):
                    planed_path_W[i,:] = np.matmul(TF, np.concatenate( (planed_path_O[i,:], [1]) ))[0:3]
                RMSE_trajectory_n_O = np.sqrt( np.sum((true_path_O - planed_path_O)**2) / (n_handover+1) )
                RMSE_trajectory_n_W = np.sqrt( np.sum((true_path_W - planed_path_W)**2) / (n_handover+1) )
                # Max velocity
                if ROBOT_ROLE == 'Giver':
                    robot_vel_idx = [dimensions.index('giver_dx'), dimensions.index('giver_dy'), dimensions.index('giver_dz')]
                    robot_acc_idx = [dimensions.index('giver_ddx'), dimensions.index('giver_ddy'), dimensions.index('giver_ddz')]
                if ROBOT_ROLE == 'Taker':
                    robot_vel_idx = [dimensions.index('taker_dx'), dimensions.index('taker_dy'), dimensions.index('taker_dz')]
                    robot_acc_idx = [dimensions.index('taker_ddx'), dimensions.index('taker_ddy'), dimensions.index('taker_ddz')]
                
                # Record handover over time performance
                location_error_over_time_W[n] = Handover_location_error_n_W
                location_error_over_time_O[n] = Handover_location_error_n_O
                time_error_over_time[n]     = Handover_time_error_n
                max_vel_over_time[n] = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover+1, robot_vel_idx]**2, 1 )))
                max_acc_over_time[n] = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover+1, robot_acc_idx]**2, 1 )))
                RMSE_over_time_O[n] = RMSE_trajectory_n_O
                RMSE_over_time_W[n] = RMSE_trajectory_n_W

                print(f'K = {problem.get_K(n):}')
                if (SAVE or SHOW):
                    plot_trajectory2(phi_nnf, n , trajectory , dt , ROBOT_ROLE , close_flag=False, true_human_x = [x_h_n_O[t_0 + i] for i in range(phi.horizon+1)], true_human_y = [y_h_n_O[t_0 + i] for i in range(phi.horizon+1)], true_human_zG = [z_h_n_O[t_0 + i] for i in range(phi.horizon+1)] )
                if SAVE:
                    plt.savefig(SAVE_PATH + f'tstart{t_0:}_n{n:}_t{t:}.png' )
                if SHOW:
                    plt.show()
                plt.close()

                # for debug only
                minK_result = problem.get_minimum_K()

                # Check if handover succeeded
                if problem.get_K(n) < success_condition: # KSHIRSAGAR_EPSILON: # [m]
                    state_machine = 3 # Success (Reset state)

                # Check if handover timeout
                if n >= phi.horizon - 1:
                    print('timeout')
                    state_machine = 2 # Fail

                # Check if handover failed
                fail_condition = not solution_found # To be decided
                if fail_condition:
                    print('path not found')
                    # Check why
                    # How high is max acceleration?
                    # How high is max speed?
                    max_acc = problem.get_max_acceleration([a for a in range(n+1)], human_pos_idx)

                    state_machine = 2 # Fail (Reset state)

    
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
                x_true_human_list.append( x_h_n_O[t] )
                y_true_human_list.append( y_h_n_O[t] )
                z_true_human_list.append( z_h_n_O[t] )
                x_true_robot_list.append( x_r_n_O[t] )
                y_true_robot_list.append( y_r_n_O[t] )
                z_true_robot_list.append( z_r_n_O[t] )
        
                # Exit planner after handover window
                if n > phi.horizon:
                    break
        
                print(f't = {t:}*dt = {t*dt:02}; t_0 = t_{t_0:} = {t_0*dt:02}; n = {n:}; state = {state_machine:}')

        # Get statistics for final plan (closed loop)
        # Success
        if state_machine == 3:
            Handover_successfull_final_plan = 1
            # Handover location
            Handover_location_final_plan = np.array( [ trajectory[n_handover][robot_pos_idx[0]], trajectory[n_handover][robot_pos_idx[1]], trajectory[n_handover][robot_pos_idx[2]] ] )
            # Handover time
            Handover_time_final_plan = (n_handover + t_0) * dt 
            # Handover location accuracy
            Handover_location_error_final_plan = np.sum( (True_giver_handover_location_O - Handover_location_final_plan)**2 )
            # Handover time accuracy
            Handover_time_error_final_plan = Handover_time_final_plan - Handover_time 
            # RMSE trajectory
            true_path = np.array([x_r_n_W[t_0:t_0+n_handover+1], y_r_n_W[t_0:t_0+n_handover+1], z_r_n_W[t_0:t_0+n_handover+1]]).T
            planed_path = np.array(trajectory)[0:n_handover+1, robot_pos_idx]
            RMSE_trajectory_final_plan = np.sqrt( np.sum((true_path-planed_path)**2) / (n_handover+1) )
            # Max velocity
            if ROBOT_ROLE == 'Giver':
                robot_vel_idx = [dimensions.index('giver_dx'), dimensions.index('giver_dy'), dimensions.index('giver_dz')]
                robot_acc_idx = [dimensions.index('giver_ddx'), dimensions.index('giver_ddy'), dimensions.index('giver_ddz')]
            if ROBOT_ROLE == 'Taker':
                robot_vel_idx = [dimensions.index('taker_dx'), dimensions.index('taker_dy'), dimensions.index('taker_dz')]
                robot_acc_idx = [dimensions.index('taker_ddx'), dimensions.index('taker_ddy'), dimensions.index('taker_ddz')]
            Handover_max_vel_final_plan = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover+1, robot_vel_idx]**2, 1 )))
            Handover_max_acc_final_plan = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover+1, robot_acc_idx]**2, 1 )))

        if state_machine == 2:
            Handover_successfull_final_plan = 0
            # Handover location (Not applicable)
            Handover_location_final_plan = [None, None, None]
            # Handover time (Not applicable)
            Handover_time_final_plan = None
            # Handover location accuracy (Not applicable)
            Handover_location_error_final_plan = None
            # Handover time accuracy (Not applicable)
            Handover_time_error_final_plan = None
            # RMSE trajectory (Remove final time step, since the planner didn't succeede it violates constraints and is not implemented)
            true_path = np.array([x_r_n_W[t_0:t_0+n_handover], y_r_n_W[t_0:t_0+n_handover], z_r_n_W[t_0:t_0+n_handover]]).T
            planed_path = np.array(trajectory)[0:n_handover, robot_pos_idx]
            RMSE_trajectory_final_plan = np.sqrt( np.sum((true_path-planed_path)**2) / (n_handover) )
            # Max velocity (Remove final time step, since the planner didn't succeede it violates constraints and is not implemented)
            if ROBOT_ROLE == 'Giver':
                robot_vel_idx = [dimensions.index('giver_dx'), dimensions.index('giver_dy'), dimensions.index('giver_dz')]
                robot_acc_idx = [dimensions.index('giver_ddx'), dimensions.index('giver_ddy'), dimensions.index('giver_ddz')]
            if ROBOT_ROLE == 'Taker':
                robot_vel_idx = [dimensions.index('taker_dx'), dimensions.index('taker_dy'), dimensions.index('taker_dz')]
                robot_acc_idx = [dimensions.index('taker_ddx'), dimensions.index('taker_ddy'), dimensions.index('taker_ddz')]
            Handover_max_vel_final_plan = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover, robot_vel_idx]**2, 1 )))
            Handover_max_acc_final_plan = np.max( np.sqrt( np.sum( np.array(trajectory)[0:n_handover, robot_acc_idx]**2, 1 )))


        # Log handover statistics to csv
        #['Handover', 
        #         'Location_x_O_1st', 'Location_y_O_1st', 'Location_z_O_1st', 
        #         'Location_error_O_1st', 
        #         'Time_1st', 'Time_error_1st', 
        #         'Max_vel_1st', 'Max_acc_1st',
        #         'RMSE_trajectory_1st',
        #         'Success_final', 
        #         'Location_x_final', 'Location_y_final', 'Location_z_final', 
        #         'Location_error_final', 
        #         'Time_final', 'Time_error_final',
        #         'Max_vel_final', 'Max_acc_final',
        #         'RMSE_trajectory_final'
        #         ]
        result_data = [RECORDED_DATA_PATH, # <- filename
                       Handover_location_1st_plan_O[0], Handover_location_1st_plan_O[1], Handover_location_1st_plan_O[2], 
                       Handover_location_error_1st_plan_O, 
                       Handover_time_1st_plan, Handover_time_error_1st_plan, 
                       Handover_max_vel_1st_plan, Handover_max_acc_1st_plan, 
                       RMSE_trajectory_1st_plan_O,
                       Handover_successfull_final_plan, 
                       Handover_location_final_plan[0], Handover_location_final_plan[1], Handover_location_final_plan[2],
                       Handover_location_error_final_plan, 
                       Handover_time_final_plan, Handover_time_error_final_plan,
                       Handover_max_vel_final_plan, Handover_max_acc_final_plan, 
                       RMSE_trajectory_final_plan
                       ]
        with open(result_filename, 'a') as file:
            result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
            result_writer.writerow(result_data)
     
        with open(result_location_filename_O, 'a') as file:
            result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
            result_writer.writerow(np.concatenate( ( [RECORDED_DATA_PATH, Handover_successfull_final_plan], location_error_over_time_O) ) )
        with open(result_location_filename_W, 'a') as file:
            result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
            result_writer.writerow(np.concatenate( ( [RECORDED_DATA_PATH, Handover_successfull_final_plan], location_error_over_time_W) ) )

        with open(result_time_filename, 'a') as file:
            result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
            result_writer.writerow(np.concatenate( ( [RECORDED_DATA_PATH, Handover_successfull_final_plan], time_error_over_time) ) )

        with open(result_max_vel_filename , 'a') as file:
            result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
            result_writer.writerow(np.concatenate( ( [RECORDED_DATA_PATH, Handover_successfull_final_plan], max_vel_over_time) ) )
        with open(result_max_acc_filename , 'a') as file:
            result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
            result_writer.writerow(np.concatenate( ( [RECORDED_DATA_PATH, Handover_successfull_final_plan], max_acc_over_time) ) )

        with open(result_RMSE_filename_O , 'a') as file:
            result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
            result_writer.writerow(np.concatenate( ( [RECORDED_DATA_PATH, Handover_successfull_final_plan], RMSE_over_time_O) ) )
        with open(result_RMSE_filename_W , 'a') as file:
            result_writer = csv.writer(file, dialect='excel', lineterminator='\n')
            result_writer.writerow(np.concatenate( ( [RECORDED_DATA_PATH, Handover_successfull_final_plan], RMSE_over_time_W) ) )
            

        #x_human_chest_list
        #y_human_chest_list
        #z_human_chest_list
        #x_robot_chest_list
        #y_robot_chest_list
        #z_robot_chest_list
        #x_robot_in_mid
        #y_robot_in_mid
        #z_robot_in_mid
        #x_human_in_mid
        #y_human_in_mid
        #z_human_in_mid

        #save_plot_recorded_handovers(dt, ROBOT_ROLE, 
        #                             x_human_in_mid, 
        #                             y_human_in_mid, 
        #                             z_human_in_mid, 
        #                             x_robot_in_mid, 
        #                             y_robot_in_mid, 
        #                             z_robot_in_mid, 
        #                             x_human_chest_list, 
        #                             y_human_chest_list, 
        #                             z_human_chest_list,
        #                             x_robot_chest_list, 
        #                             y_robot_chest_list, 
        #                             z_robot_chest_list,
        #                             predicted_handover_n_list,
        #                             predicted_handover_K_list,
        #                             trajectory_list,
        #                             dimensions,
        #                             x_true_human_list,
        #                             y_true_human_list,
        #                             z_true_human_list,
        #                             x_true_robot_list,
        #                             y_true_robot_list,
        #                             z_true_robot_list
        #                             )


        #with open('x_human_in_mid.csv', 'w') as f:
        #    for d in x_human_in_mid:
        #        f.write(str(d))
        #        f.write("\n")

        #with open('y_human_in_mid.csv', 'w') as f:
        #    for d in y_human_in_mid:
        #        f.write(str(d))
        #        f.write("\n")

        #with open('z_human_in_mid.csv', 'w') as f:
        #    for d in z_human_in_mid:
        #        f.write(str(d))
        #        f.write("\n")

        #with open('x_rob_in_mid.csv', 'w') as f:
        #    for d in x_robot_in_mid:
        #        f.write(str(d))
        #        f.write("\n")

        #with open('y_rob_in_mid.csv', 'w') as f:
        #    for d in y_robot_in_mid:
        #        f.write(str(d))
        #        f.write("\n")

        #with open('z_rob_in_mid.csv', 'w') as f:
        #    for d in z_robot_in_mid:
        #        f.write(str(d))
        #        f.write("\n")

        #with open('human_chest_list.csv', 'w') as f:
        #    for d in human_chest_list:
        #        f.write(str(d))
        #        f.write("\n")

        #with open('robot_chest_list.csv', 'w') as f:
        #    for d in robot_chest_list:
        #        f.write(str(d))
        #        f.write("\n")