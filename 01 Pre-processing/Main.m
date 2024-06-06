% Learning data generator main

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-process handover data for 
% Thesis: Automated Control of Human-Robot Handovers using Data-driven STL Modeling of Human-Human Handovers
% Author: Jonathan Fredberg; jfredb@kth.se

% Matlab R2023b
% Requires Toolboxes/built in functions:
% quat2tform (Robotics System Toolbox)
% downsample (Signal Processing Toolbox)

% All settings and tunable parameters are available in "%% Set parameters"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
%% Set parameters

% Input dataset filepaths
% % Small dataset (EXAMPLE)
INPUT_DATASET.folders = ["D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair7\Setting1\Set1\handovers_P7_S1"]
INPUT_DATASET.name = ["P7_S1"]

% Large dataset (EXAMPLE)
INPUT_DATASET.folders = ["D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair1\Setting1\Set1\handovers_P1_S1_new", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair1\Setting1\Set2\handovers_P1_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair2\Setting1\Set1\handovers_P2_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair2\Setting1\Set2\handovers_P2_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair3\Setting1\Set1\handovers_P3_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair3\Setting1\Set2\handovers_P3_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair4\Setting1\Set1\handovers_P4_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair4\Setting1\Set2\handovers_P4_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair5\Setting1\Set1\handovers_P5_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair5\Setting1\Set2\handovers_P5_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair6\Setting1\Set1\handovers_P6_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair6\Setting1\Set2\handovers_P6_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair7\Setting1\Set1\handovers_P7_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair7\Setting1\Set2\handovers_P7_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair8\Setting1\Set1\handovers_P8_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair8\Setting1\Set2\handovers_P8_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair9\Setting1\Set1\handovers_P9_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair9\Setting1\Set2\handovers_P9_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair10\Setting1\Set1\handovers_P10_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair10\Setting1\Set2\handovers_P10_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair11\Setting1\Set1\handovers_P11_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair11\Setting1\Set2\handovers_P11_S2", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair12\Setting1\Set1\handovers_P12_S1", ...
    "D:\Electrical Engineering\Offline Exjobb\Handover dataset\dataset-main\dataset_pairwise\Pair12\Setting1\Set2\handovers_P12_S2"];
INPUT_DATASET.name = ["P1_S1", "P1_S2", "P2_S1", "P2_S2", "P3_S1", "P3_S2", "P4_S1", "P4_S2", "P5_S1", "P5_S2", ...
    "P6_S1", "P6_S2", "P7_S1", "P7_S2", "P8_S1", "P8_S2", "P9_S1", "P9_S2", "P10_S1", "P10_S2", "P11_S1", "P11_S2", "P12_S1", "P12_S2"];


% Output data set
LEARNING_DATA_FILEPATH.folder = "Output_Dataset_Learning\"
VALIDATION_DATA_FILEPATH.folder = "Output_Dataset_Validation\"
MAT_DATASET.filename = "Output_dataset.mat";


% Validation data ratio
LEARNING_TO_VALIDATION_RATIO = 0.2; % validation on a random 20% of all data

% Outlier detection settings
OUTLIER_SETTINGS.Start_t = -2;      % Start time for relevant data
OUTLIER_SETTINGS.End_t = 2;         % End time for relevant data
OUTLIER_SETTINGS.std_tol = 2.5;     % Signal must not deviate from the mean...
OUTLIER_SETTINGS.samples_tol = 40;  % for more than this many samples

% Pickup zone detection setting
    % Pickup zone is a radius around the location of giver_right_hand 
    % when they first grab the object. Handover starts when 
    % giver_right_hand moves out from the pickup zone.
PICKUP_ZONE_RADIUS = 0.1;           % Radius of pickup zone 
    
% Object ownership settings
OWNERSHIP.grip_threshold = 1;       % Force threshold detecting when giver/taker is holding the object (Newtons)

% Downsample
DOWNSAMPLE.downsample_factor = 5;   % Determines sample rate of output data
    % Input sample rate (120Hz) / downsample_factor = Output samle rate

% Filtering settings
    % Estimates position, velocity and acceleration using tuned kalman filter, 
    % And removes handovers with tracking error above threshold.
SMOOTHING_SETTINGS.error_threshold = 0.08   % Max tracking error (Meter)
SMOOTHING_SETTINGS.kalman_alpha = 50        % Kalman filter tuning parameter

%% Set Output traces
clear OUTPUT_SIGNALS
% OUTPUT_SIGNALS(1).name = "x_baton_c2c";
% OUTPUT_SIGNALS(1).frame = "c2c";
% OUTPUT_SIGNALS(1).type = "Position";
% OUTPUT_SIGNALS(1).trace = "baton";
% OUTPUT_SIGNALS(1).axis = 1;
% 
% OUTPUT_SIGNALS(2).name = "y_giver_RHand_c2c";
% OUTPUT_SIGNALS(2).frame = "c2c";
% OUTPUT_SIGNALS(2).type = "Position";
% OUTPUT_SIGNALS(2).trace = "giver_RHand"
% OUTPUT_SIGNALS(2).axis = 2;
% 
% OUTPUT_SIGNALS(3).name = "vz_baton_rh2rh";
% OUTPUT_SIGNALS(3).frame = "rh2rh";
% OUTPUT_SIGNALS(3).type = "Velocity";
% OUTPUT_SIGNALS(3).trace = "baton"
% OUTPUT_SIGNALS(3).axis = 3;
% 
% OUTPUT_SIGNALS(4).name = "o_g";
% OUTPUT_SIGNALS(4).frame = "ownership";
% OUTPUT_SIGNALS(4).type = "Ownership";
% OUTPUT_SIGNALS(4).trace = "giver_owner"
% OUTPUT_SIGNALS(4).axis = 1;
% 
% OUTPUT_SIGNALS(5).name = "F_g";
% OUTPUT_SIGNALS(5).frame = "forces";
% OUTPUT_SIGNALS(5).type = "grip";% ???;
% OUTPUT_SIGNALS(5).trace = "taker_grip"
% OUTPUT_SIGNALS(5).axis = 1;

% OUTPUT_SIGNALS(1).name = "x_baton";
% OUTPUT_SIGNALS(1).frame = "c2c";
% OUTPUT_SIGNALS(1).type = "Position";
% OUTPUT_SIGNALS(1).trace = "baton"
% OUTPUT_SIGNALS(1).axis = 1;
% 
% OUTPUT_SIGNALS(end+1).name = "y_baton";
% OUTPUT_SIGNALS(end).frame = "c2c";
% OUTPUT_SIGNALS(end).type = "Position";
% OUTPUT_SIGNALS(end).trace = "baton"
% OUTPUT_SIGNALS(end).axis = 2;
% 
% OUTPUT_SIGNALS(end+1).name = "z_baton";
% OUTPUT_SIGNALS(end).frame = "c2c";
% OUTPUT_SIGNALS(end).type = "Position";
% OUTPUT_SIGNALS(end).trace = "baton"
% OUTPUT_SIGNALS(end).axis = 3;
% 
% OUTPUT_SIGNALS(end+1).name = "dx_baton";
% OUTPUT_SIGNALS(end).frame = "c2c";
% OUTPUT_SIGNALS(end).type = "Velocity";
% OUTPUT_SIGNALS(end).trace = "baton"
% OUTPUT_SIGNALS(end).axis = 1;
% 
% OUTPUT_SIGNALS(end+1).name = "dy_baton";
% OUTPUT_SIGNALS(end).frame = "c2c";
% OUTPUT_SIGNALS(end).type = "Velocity";
% OUTPUT_SIGNALS(end).trace = "baton"
% OUTPUT_SIGNALS(end).axis = 2;
% 
% OUTPUT_SIGNALS(end+1).name = "dz_baton";
% OUTPUT_SIGNALS(end).frame = "c2c";
% OUTPUT_SIGNALS(end).type = "Velocity";
% OUTPUT_SIGNALS(end).trace = "baton"
% OUTPUT_SIGNALS(end).axis = 3;

OUTPUT_SIGNALS(1).name = "x_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Position";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 1;

OUTPUT_SIGNALS(end+1).name = "y_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Position";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 2;

OUTPUT_SIGNALS(end+1).name = "z_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Position";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 3;

OUTPUT_SIGNALS(end+1).name = "dx_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Velocity";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 1;

OUTPUT_SIGNALS(end+1).name = "dy_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Velocity";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 2;

OUTPUT_SIGNALS(end+1).name = "dz_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Velocity";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 3;

OUTPUT_SIGNALS(end+1).name = "ddx_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Acceleration";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 1;

OUTPUT_SIGNALS(end+1).name = "ddy_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Acceleration";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 2;

OUTPUT_SIGNALS(end+1).name = "ddz_taker_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Acceleration";
OUTPUT_SIGNALS(end).trace = "taker_RHand"
OUTPUT_SIGNALS(end).axis = 3;

OUTPUT_SIGNALS(end+1).name = "x_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Position";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 1;

OUTPUT_SIGNALS(end+1).name = "y_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Position";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 2;

OUTPUT_SIGNALS(end+1).name = "z_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Position";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 3;

OUTPUT_SIGNALS(end+1).name = "dx_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Velocity";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 1;

OUTPUT_SIGNALS(end+1).name = "dy_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Velocity";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 2;

OUTPUT_SIGNALS(end+1).name = "dz_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Velocity";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 3;

OUTPUT_SIGNALS(end+1).name = "ddx_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Acceleration";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 1;

OUTPUT_SIGNALS(end+1).name = "ddy_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Acceleration";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 2;

OUTPUT_SIGNALS(end+1).name = "ddz_giver_RHand";
OUTPUT_SIGNALS(end).frame = "c2c";
OUTPUT_SIGNALS(end).type = "Acceleration";
OUTPUT_SIGNALS(end).trace = "giver_RHand"
OUTPUT_SIGNALS(end).axis = 3;

% Map dataset
clear OUTPUT_SIGNALS_map
OUTPUT_SIGNALS_map(1).name = "x_taker_RHand";
OUTPUT_SIGNALS_map(end).frame = "map";
OUTPUT_SIGNALS_map(end).type = "Position";
OUTPUT_SIGNALS_map(end).trace = "taker_RHand"
OUTPUT_SIGNALS_map(end).axis = 1;

OUTPUT_SIGNALS_map(end+1).name = "y_taker_RHand";
OUTPUT_SIGNALS_map(end).frame = "map";
OUTPUT_SIGNALS_map(end).type = "Position";
OUTPUT_SIGNALS_map(end).trace = "taker_RHand"
OUTPUT_SIGNALS_map(end).axis = 2;

OUTPUT_SIGNALS_map(end+1).name = "z_taker_RHand";
OUTPUT_SIGNALS_map(end).frame = "map";
OUTPUT_SIGNALS_map(end).type = "Position";
OUTPUT_SIGNALS_map(end).trace = "taker_RHand"
OUTPUT_SIGNALS_map(end).axis = 3;

OUTPUT_SIGNALS_map(end+1).name = "x_giver_RHand";
OUTPUT_SIGNALS_map(end).frame = "map";
OUTPUT_SIGNALS_map(end).type = "Position";
OUTPUT_SIGNALS_map(end).trace = "giver_RHand"
OUTPUT_SIGNALS_map(end).axis = 1;

OUTPUT_SIGNALS_map(end+1).name = "y_giver_RHand";
OUTPUT_SIGNALS_map(end).frame = "map";
OUTPUT_SIGNALS_map(end).type = "Position";
OUTPUT_SIGNALS_map(end).trace = "giver_RHand"
OUTPUT_SIGNALS_map(end).axis = 2;

OUTPUT_SIGNALS_map(end+1).name = "z_giver_RHand";
OUTPUT_SIGNALS_map(end).frame = "map";
OUTPUT_SIGNALS_map(end).type = "Position";
OUTPUT_SIGNALS_map(end).trace = "giver_RHand"
OUTPUT_SIGNALS_map(end).axis = 3;



% OUTPUT_SIGNALS(end+1).name = "z_rh";
% OUTPUT_SIGNALS(end).frame = "rh2rh";
% OUTPUT_SIGNALS(end).type = "Position";
% OUTPUT_SIGNALS(end).trace = "baton"
% OUTPUT_SIGNALS(end).axis = 3;

% OUTPUT_SIGNALS(end+1).name = "o_g";
% OUTPUT_SIGNALS(end).frame = "ownership";
% OUTPUT_SIGNALS(end).type = "Ownership";
% OUTPUT_SIGNALS(end).trace = "giver_owner"
% OUTPUT_SIGNALS(end).axis = 1;
% 
% OUTPUT_SIGNALS(end+1).name = "o_s";
% OUTPUT_SIGNALS(end).frame = "ownership";
% OUTPUT_SIGNALS(end).type = "Ownership";
% OUTPUT_SIGNALS(end).trace = "object_shared"
% OUTPUT_SIGNALS(end).axis = 1;
% 
% OUTPUT_SIGNALS(end+1).name = "e";
% OUTPUT_SIGNALS(end).frame = "ownership";
% OUTPUT_SIGNALS(end).type = "Ownership";
% OUTPUT_SIGNALS(end).trace = "handover_trigger"
% OUTPUT_SIGNALS(end).axis = 1;

% Coordinate frame settings
% Field names must align with frames of the ouput signals
% * Origin of frames is the midpoint between point1 and point2
% * z-axis is vertical, aligned with map z-axis
% * x-axis points from a vertical line through point1 to vertical line
%       through point2
% * y-axis is defined perpendicular to x and z axis
FRAMES.c2c.name = "chest to chest";
FRAMES.c2c.point1 = "giver_chest";
FRAMES.c2c.point2 = "taker_chest";

FRAMES.h2h.name = "head to head";
FRAMES.h2h.point1 = "giver_head";
FRAMES.h2h.point2 = "taker_head";

FRAMES.rh2rh.name = "right hand to right hand";
FRAMES.rh2rh.point1 = "giver_RHand";
FRAMES.rh2rh.point2 = "taker_RHand";



%% Process settings

% Get all relevant traces
all_trace_keys = ["giver_grip", "taker_grip"];
all_frames = string([]);

for idx = 1:length(OUTPUT_SIGNALS)
    all_trace_keys(end+1) = OUTPUT_SIGNALS(idx).trace;
    all_frames(end+1) = OUTPUT_SIGNALS(idx).frame;
end

all_frames = unique(all_frames);
all_frames = setdiff(all_frames, ["local","forces","ownership"]);
for field = all_frames
    all_trace_keys(end+1) = FRAMES.(field).point1; % Add points defining shared frame to be processed
    all_trace_keys(end+1) = FRAMES.(field).point2;
end
all_trace_keys = unique(all_trace_keys);
all_trace_keys = setdiff(all_trace_keys, ["", "giver_owner", "taker_owner", "object_shared", "handover_trigger"]);

%% Load data

[handover_set, N_sets] = load_datasets(INPUT_DATASET, all_trace_keys);

% load('C:\Users\jfred\FlaoGDrive\Exjobb Drive folder\Exjobb\Learning_data_generator\Datasets\dataset_all_pairs_map_unfiltered.mat')
% handover_set = dataset_map_unfiltered;

%% Count handovers!!!
Total_handover_count = 0;
for sets = handover_set
    Total_handover_count = Total_handover_count + sets.N_handovers;
end
Total_handover_count

%% Manually remove problem signals
set_name = "P4_S1";
handover_name = 'handover_20';
handover_set = remove_handover(handover_set, set_name, handover_name);

set_name = "P5_S1";
handover_name = "handover_28";
handover_set = remove_handover(handover_set, set_name, handover_name);
       
set_name = "P5_S2";
handover_name = "handover_51";
handover_set = remove_handover(handover_set, set_name, handover_name);

%% Count handovers!!!
Handover_count_after_manual_removal = 0;
for sets = handover_set
    Handover_count_after_manual_removal = Handover_count_after_manual_removal + sets.N_handovers;
end
Handover_count_after_manual_removal


%% Smooth data and estimate speed AND Remove outliers with tracking error above threshold
% Estimates velocity AND acceleration 
% And removes handovers with tracking error above threshold, filtering out
%   signals with tracking errors


handover_set = extimate_pos_vel_acc(handover_set, SMOOTHING_SETTINGS.error_threshold, SMOOTHING_SETTINGS.kalman_alpha);

%% Count handovers!!!
Handover_count_after_tracking_error = 0;
for sets = handover_set
    Handover_count_after_tracking_error = Handover_count_after_tracking_error + sets.N_handovers;
end
Handover_count_after_tracking_error


%% Transform to new coordinates 

handover_set = transform_set(handover_set, FRAMES, all_frames, all_trace_keys);


%% Detect object ownership timeing

handover_set = detect_ownership_set(handover_set, OWNERSHIP, false);

%% Detect pick up zone 


handover_set = detect_pickup_zone(handover_set, PICKUP_ZONE_RADIUS);

%% Trim unwanted data, and adjust time scale, downsample
% Start when giver_RHand leaves the pick-up zone
% End at last sample of taker_owner = 1

handover_set = trim_handovers_pickup_zone(handover_set, DOWNSAMPLE);

%% Pad handovers
handover_set = pad_handovers_zero_acc_vel(handover_set);

%% Remove outliers 

% Currently only filtering ALL USED TRACES, on ALL AXIS.
for set_idx = 1:N_sets
    fprintf("\nFiltering %d handovers from " + handover_set(set_idx).name + "\n", handover_set(set_idx).N_handovers)

    N_before = handover_set(set_idx).N_handovers;

    [handover_set(set_idx).handover, inlier_idx, outlier_idx] = filter_handover_set_after_trim(handover_set(set_idx).handover, OUTLIER_SETTINGS.std_tol, OUTLIER_SETTINGS.samples_tol, "Trace_keys", all_trace_keys, "Frame", ["map", "forces"]);
    rejected = length(outlier_idx);
    handover_set(set_idx).N_handovers = length(inlier_idx);
    % handover_set(set_idx).name = "INLIERS " + handover_set(set_idx).name;
    rejection_ratio = rejected / N_before;

    fprintf("%2.0f%% handovers rejected from set " + handover_set(set_idx).name + "\n", rejection_ratio*100)
end

%% Count handovers!!!
Handover_count_after_outlier_rejection = 0;
for sets = handover_set
    Handover_count_after_outlier_rejection = Handover_count_after_outlier_rejection + sets.N_handovers;
end
Handover_count_after_outlier_rejection

%% Tally handovers
% Total_handover_count
% Handover_count_after_manual_removal
% Handover_count_after_tracking_error
% Handover_count_after_outlier_rejection
count_struct.Name = ["Total from dataset";
    "Manually remover"; "Filtered for tracking errors";
    "Filtered as outliers"; "Final total"];
count_struct.Values = [Total_handover_count;
    Total_handover_count - Handover_count_after_manual_removal;
    Handover_count_after_manual_removal - Handover_count_after_tracking_error;
    Handover_count_after_tracking_error - Handover_count_after_outlier_rejection;
    Handover_count_after_outlier_rejection];

handover_count_table = struct2table(count_struct)
%% Devide dataset in learning and validation data
% error("I don't wanna accidentally overwrite datasets")
clear validation_set
clear learning_set

for set_idx = 1:N_sets
    N_handovers = handover_set(set_idx).N_handovers;

    N_validation = round(N_handovers * LEARNING_TO_VALIDATION_RATIO);
    N_learning = N_handovers - N_validation;

    validation_idx = randperm(N_handovers, N_validation);
    learning_idx = setdiff([1:N_handovers], validation_idx);

    validation_set(set_idx).handover(1:N_validation) = handover_set(set_idx).handover(validation_idx);
    validation_set(set_idx).name = handover_set(set_idx).name + "_validation";
    validation_set(set_idx).N_handovers = N_validation;


    learning_set(set_idx).handover(1:N_learning) = handover_set(set_idx).handover(learning_idx);
    learning_set(set_idx).name = handover_set(set_idx).name;
    learning_set(set_idx).N_handovers = N_learning;

    handover_set(set_idx).validation_idx = validation_idx;
    handover_set(set_idx).learning_idx = learning_idx;
end

%% Write datasets
% error("I don't wanna accidentally overwrite datasets")
learning_set = write_dataset(learning_set, OUTPUT_SIGNALS, LEARNING_DATA_FILEPATH);
validation_set = write_dataset(validation_set, OUTPUT_SIGNALS, VALIDATION_DATA_FILEPATH);

save(MAT_DATASET.filename, "validation_set", "learning_set", "handover_set")

%%
% dummy_path = LEARNING_DATA_FILEPATH;
% dummy_path.folder = dummy_path.folder + "DUMMY"
% dataset_map_unfiltered = write_dataset(handover_set, OUTPUT_SIGNALS_map, dummy_path );
% save('dataset_map_unfiltered.mat', "dataset_map_unfiltered")