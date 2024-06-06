function [handover_set, N_sets] = load_datasets(INPUT_DATASET, Trace_keys)
% Load the sets defined in INPUT_DATASET, only load traces defined in
% Trace_keys

% Load datasets
set_idx = 0;
for set_path = INPUT_DATASET.folders
    set_idx = set_idx+1;
    handovers = load_handover_set(set_path, "Trace_keys", Trace_keys);
    handover_set(set_idx).handover = handovers;
    handover_set(set_idx).N_handovers = length(handovers);
    handover_set(set_idx).name = INPUT_DATASET.name(set_idx);
end

% Output number of sets
N_sets = set_idx;

end % Function end


function [handovers] = load_handover_set(filepath, varargin)
%[handovers] = load_handover_set(filepath, "Trace_keys", "All_traces")
%   Detailed explanation goes here

p = inputParser;
addParameter(p,"Trace_keys",["baton_pose", "giver_RHand", "taker_RHand", "giver_chest", "taker_chest"]);
addParameter(p,"All_traces", false, @islogical);

parse(p,varargin{:})
Trace_keys = p.Results.Trace_keys;

if p.Results.All_traces
    Trace_keys = load('All_trace_keys.mat',"All_trace_keys").All_trace_keys;
end

%% Loop through all handovers
filepath = replace(filepath, "\", "/");
set_dir = dir(filepath);
N = length(set_dir);

fprintf("\nLoading handovers from " + filepath + "\n")

for folder_idx = 1:N
    loop_start = tic;
    folder_name = set_dir(folder_idx).name;
    handover_idx = str2double(erase(folder_name,"handover_"));

    if isscalar(handover_idx) & contains(folder_name, "handover_")
        handover = load_handover(filepath + "/" + folder_name + "/", "Trace_keys", Trace_keys);
        handover.name = folder_name;
        handovers(handover_idx) = handover;
        
        loop_time = toc(loop_start);
        if mod(folder_idx-3,20) == 0
            % fprintf(estimated_time_left(loop_time, N-folder_idx))
        end
    end
end

fprintf("\n%d handovers loaded from " + filepath + "\n", length(handovers))

end % function end


function handover = load_handover(folder, varargin)
% handover = load_handover(folder, Trace_keys, All_traces = true

%   Loads from csv files from correctly formated folder
% Input:
%   folder      string
% Output:
%   handover                    struct
%       .name                   string
%       .Fs                     double
%       .signals                struct
%           .[trace name]       struct
%               .name           string
%               .frame          string
%               .type           string
%               .trace          string
%               .data           array   (1xN 3xN double or 1xN quaternion)

N = 801;
handover.Fs = 120;
handover.name = folder;

% time trace
handover.signals.time.name = "time";
handover.signals.time.type = "time";
handover.signals.time.data = [-400/handover.Fs: 1/handover.Fs : 400/handover.Fs];

Fs = 120;
t = [-400/Fs: 1/Fs : 400/Fs]';

p = inputParser;
addParameter(p,"Trace_keys",["baton_pose", "giver_RHand", "taker_RHand", "giver_chest", "taker_chest"]);
addParameter(p,"All_traces", false, @islogical);

parse(p,varargin{:})
Trace_keys = p.Results.Trace_keys;

if p.Results.All_traces
    Trace_keys = load('All_trace_keys.mat',"All_trace_keys").All_trace_keys;
end
for key = Trace_keys
    if ((key == "interaction_force") || (key == "interaction_torque"))
        % Load Interaction wrench
        [Interaction_force, Interaction_torque] = load_wrench(folder + "Wrench_interaction_saved.csv");
        handover.signals.forces.interaction_force.data  = Interaction_force;
        handover.signals.forces.interaction_force.name  = "interaction_force";
        handover.signals.forces.interaction_force.frame = "local";
        handover.signals.forces.interaction_force.type  = "force"
        handover.signals.forces.interaction_force.trace = "";
        
        handover.signals.forces.interaction_torque.data  = Interaction_torque;
        handover.signals.forces.interaction_torque.name  = "interaction_torque";
        handover.signals.forces.interaction_torque.frame = "local";
        handover.signals.forces.interaction_torque.type  = "torque"
        handover.signals.forces.interaction_torque.trace = "";
    
    % Load Grip forces
    elseif (key == "giver_grip")
        % handover.signals.giver_grip.data = zeros(1,N);
        handover.signals.forces.giver_grip.name = "giver_grip";
        handover.signals.forces.giver_grip.frame = "local";
        handover.signals.forces.giver_grip.type = "grip";
        handover.signals.forces.giver_grip.trace = "";

        Giver_Force = load_wrench(folder + "Wrench_giver_saved.csv");
        handover.signals.forces.giver_grip.data = -Giver_Force(:,3);
    
    elseif (key == "taker_grip")
        % handover.signals.taker_grip.data = zeros(1,N);
        handover.signals.forces.taker_grip.name = "taker_grip";
        handover.signals.forces.taker_grip.frame = "local";
        handover.signals.forces.taker_grip.type = "grip";
        handover.signals.forces.taker_grip.trace = "";

        Taker_Force = load_wrench(folder + "Wrench_taker_saved.csv");
        handover.signals.forces.taker_grip.data = -Taker_Force(:,3);
    else
    
    
        % Position and orientation traces
        [handover.signals.map.(key).data, handover.signals.map.(key+"_orientation").data] = load_object(folder + key + "_pose_saved.csv");

        handover.signals.map.(key).name     = key;
        handover.signals.map.(key).frame    = "map";
        handover.signals.map.(key).type     = "position";
        handover.signals.map.(key).trace    = key;

        handover.signals.map.(key + "_orientation").name     = key + "_orientation";
        handover.signals.map.(key + "_orientation").frame    = "map";
        handover.signals.map.(key + "_orientation").type     = "orientation";
        handover.signals.map.(key + "_orientation").trace    = key;

    end
end

% function end
end


function [Position, Orientation] = load_object(filename)
% [Position, Orientation] = load_object(filename)
%   Detailed explanation goes here

% load csv
M = readmatrix(filename);

Position = M(:,1:3);
Orientation = quaternion(M(:,4:7));
end

function [Force, Torque] = load_wrench(filename)
% [Force, Torque] = load_wrench(filename)
%   Detailed explanation goes here

% load csv
M = readmatrix(filename);

Force = M(:,1:3);
Torque = M(:,4:6);
end