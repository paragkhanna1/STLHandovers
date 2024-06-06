function [new_frame] = p2p_Basili_frame(start_frame, point1_key, point2_key, varargin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Parse optional input
p = inputParser;
addParameter(p,"Trace_keys","empty");
addParameter(p,"All_traces", true, @islogical);

parse(p,varargin{:})

if p.Results.All_traces && (contains("empty", p.Results.Trace_keys))
    Trace_keys = load('All_trace_keys.mat',"All_trace_keys").All_trace_keys;
else
    Trace_keys = p.Results.Trace_keys;
end
%% 

N_t = length(start_frame.(point1_key + "_orientation").data); % time length

% Define variables outside loop
point1_pos = start_frame.(point1_key).data;
point2_pos = start_frame.(point2_key).data;
origin_pos = (point2_pos + point1_pos)./2; % Origin point over time in old frame
ez = repmat([0 0 1],N_t,1); % z base vector (same in both frames)
ex = [point2_pos(:,1:2) - point1_pos(:,1:2), zeros(N_t, 1)];
ey = cross(ez, ex);

% Rotation matrix (column, row, time)
Rot_M_new_to_old = zeros(3,3,N_t);
Rot_M_new_to_old(:,1,:) = ex';
Rot_M_new_to_old(:,2,:) = ey';
Rot_M_new_to_old(:,3,:) = ez';

% Homogenious transformation matrix (new->old frame) (column, row, time)
TF_new_to_old = zeros(4,4,N_t);
TF_new_to_old(1:3,1:3,:) = Rot_M_new_to_old;
TF_new_to_old(1:3, 4, :) = origin_pos';
TF_new_to_old(4, 4, :) = 1;

% Invert all transformation matricies and rotation matricies (old->new frame)
for n = 1:N_t
    TF_old_to_new(:,:,n) = TF_new_to_old(:,:,n)^-1;
    Rot_M_old_to_new(:,:,n) = Rot_M_new_to_old(:,:,n)^-1;
end

% reshape Transformation matrix to transform at all times in one matrix
TF_cell = mat2cell(reshape(TF_old_to_new, 4, 4*N_t), 4, ones(1,N_t)*4);
TF_mat = blkdiag(TF_cell{:}); % Matrix with TF on the diagonal

Rot_M_cell = mat2cell(reshape(Rot_M_old_to_new, 3, 3*N_t), 3, ones(1,N_t)*3);
Rot_M_Mat = blkdiag(Rot_M_cell{:}); % Matrix with TF on the diagonal


% Create output variable
new_frame = struct();

% Loop through traces
for key = Trace_keys
    % %% Only pose
    % % Reshape old frame
    % x_old = [traces(key).Position, ones(N_t,1)]; % Homogenious position coordinates
    % x_old = reshape(x_old', N_t*4, 1);
    % 
    % % Transform traces position
    % x_new = TF_mat * x_old;
    % 
    % % Reshape new frame
    % x_new = reshape(x_new, 4, 801)';
    % 
    % % Save position for output
    % traces_new(key).Position_a = x_new(:,1:3);
    
    %% Pose and orientation
    % TF from map to trace
    TF_map_2_trace = quatpos2tform(start_frame.(key + "_orientation").data, start_frame.(key).data);
    
    % Reshape
    TF_map_2_trace_cell = mat2cell(reshape(TF_map_2_trace, 4, 4*N_t), 4, ones(1,N_t)*4)';
    TFrs_map_2_trace = cell2mat(TF_map_2_trace_cell);

    
    % Multiply
    clear TF_new_2_trace
    TF_new_2_trace = TF_mat * TFrs_map_2_trace;
   
    % Reshape
    TF_new_2_trace = mat2cell(TF_new_2_trace, ones(1,N_t)*4, 4)'; % Transpose Nx1 -> 1xN cells
    TF_new_2_trace = reshape(cell2mat(TF_new_2_trace),4,4,N_t); % Reshape 4x4*N to 4x4xN

    % Get quaternion orientation
    new_frame.(key + "_orientation").data = quaternion(tform2quat(TF_new_2_trace));
    new_frame.(key + "_orientation").type = "orientation";
    new_frame.(key + "_orientation").name = key + "_orientation";
    new_frame.(key + "_orientation").trace = key;
    
    % Get position
    new_frame.(key).data = reshape(TF_new_2_trace(1:3,4,:),3,N_t)';
    new_frame.(key).type = "position";
    new_frame.(key).name = key;
    new_frame.(key).trace = key;


    %% vel
    if isfield(start_frame, key + "_vel")
        % % Reshape old frame
        x_old = start_frame.(key + "_vel").data; % NOT Homogenious position coordinates
        x_old = reshape(x_old', N_t*3, 1);
        % 
        % % Transform traces position
        x_new = Rot_M_Mat * x_old;
        % 
        % Reshape new frame
        x_new = reshape(x_new, 3, N_t)';
        % 
        % Save position for output
        new_frame.(key + "_vel").data= x_new(:,1:3);
        new_frame.(key + "_vel").type = "velocity";
        new_frame.(key + "_vel").name = start_frame.(key + "_vel").name;
        new_frame.(key + "_vel").trace = start_frame.(key + "_vel").trace;
    end

    %% acc
    if isfield(start_frame, key + "_acc") % The acceleration due to 
        % % Reshape old frame
        x_old = start_frame.(key + "_acc").data; % NOT Homogenious position coordinates
        x_old = reshape(x_old', N_t*3, 1);
        % 
        % % Transform traces position
        x_new = Rot_M_Mat * x_old;
        % 
        % Reshape new frame
        x_new = reshape(x_new, 3, N_t)';
        % 
        % Save position for output
        new_frame.(key + "_acc").data= x_new(:,1:3);
        new_frame.(key + "_acc").type = "acceleration";
        new_frame.(key + "_acc").name = start_frame.(key + "_acc").name;
        new_frame.(key + "_acc").trace = start_frame.(key + "_acc").trace;
    end
end

    
end