function [handovers_out, inlier_idx, outlier_idx] = filter_handover_set(handovers, Start_t, End_t, std_tol, window, varargin)
%[handovers_out, inlier_idx, outlier_idx] = filter_handover_set(handovers, Start_t, End_t, std_tol, window, "Trace_keys","All_traces", "Frame")
%   Detailed explanation goes here

% Input:
%   handovers       1xN struct  contains handover data from N handovers
%       .Map        1xK Dictionary of K traces
%           .Position   801x3 double    Position over time, [x;y;z]
%   Start_t         1x1 double  -3.3 < t < 3.3
%   End_t           1x1 double  -3.3 < t < 3.3; End_t > Start_t
%   std_tol         1x1 double  How many standard deviations from the mean
%                               are acceptable
%   window          1x1 double  Number of samples violating std_tol before
%                               a handoever is rejected

% Output:
%   handovers_out   1xN struct

% For each trace
% For each timestep from Start_t seconds to End_t seconds
%   For all handovers
%       Calculate mean and variance over all handovers at each timestep
%       Flag any trace with a deviation greater than STD_TOL
%       If deviation continues for longer than WINDOW that handover is
%       rejected as an outlier

p = inputParser;
addParameter(p,"Trace_keys",["baton_pose", "giver_RHand", "taker_RHand", "giver_chest", "taker_chest"]);
addParameter(p,"All_traces", false, @islogical);
addParameter(p,"Frame", "map", @isstring);

parse(p,varargin{:})
Trace_keys = p.Results.Trace_keys;
Frames = p.Results.Frame;

if p.Results.All_traces
    Trace_keys = load('All_trace_keys.mat',"All_trace_keys").All_trace_keys;
end

%% set variables for testing
% Trace_keys = load('All_trace_keys.mat',"All_trace_keys").All_trace_keys;
% handovers = handover_set(1).handover
% Start_t = -3
% End_t = 3
% std_tol = 3
% window = 10 % Require consecutive samples as outliers to reject data.
%%

N_handovers = length(handovers);

Start_idx = find(handovers(1).signals.time.data > Start_t, 1, "first");
End_idx = find(handovers(1).signals.time.data  > End_t, 1, "first");
t_len = End_idx - Start_idx + 1;
t_idx = [Start_idx : End_idx];

outlier_idx = [];
inlier_idx = 1:N_handovers;

for frame = Frames
    Trace_keys_in_frame = string(fieldnames(handovers(1).signals.(frame))');
    for key = intersect(Trace_keys_in_frame, Trace_keys)

        % Get signal type (and set number of axies)
        if (handovers(1).signals.(frame).(key).type == "position")
            axies = [1:3];
        elseif (handovers(1).signals.(frame).(key).type == "Velocity")
            axies = [1:3];
        elseif (handovers(1).signals.(frame).(key).type == "Acceleration")
            axies = [1:3];
        elseif (handovers(1).signals.(frame).(key).type == "orientation")
            error("outlier filtering on orientation not implemented")
        elseif (handovers(1).signals.(frame).(key).type == "grip")
            axies = 1;
        elseif (handovers(1).signals.(frame).(key).type == "force")
            axies = [1:3];    
        elseif (handovers(1).signals.(frame).(key).type == "torque")
            axies = [1:3];    
        end
    
        % Get the mean and variance for the trace at all times
        trace = zeros(N_handovers, t_len, length(axies));
        for handover_idx = inlier_idx
            trace(handover_idx, 1:t_len, axies) = handovers(handover_idx).signals.(frame).(key).data(t_idx,axies);
        end
        [trace_std, trace_mean] = std(trace(:,:,:), 0, 1);
    
        % Check to see which traces deviate more than var_tol allows
        trace_deviation = abs(trace - trace_mean);
        outlier_samples = trace_deviation > trace_std .* std_tol;
    
        outlier_samples = max(outlier_samples,[],3); % take max over xyz. If any are outlier, all will be discarded.
    
        % Find length of adjacent 1:s
        for handover_idx = inlier_idx
            A = outlier_samples(handover_idx, :);
            L=cumsum([1, diff(A)~=0]);
            B=splitapply(@sum,A,L);
            longest_sequential = max(B);
            if longest_sequential > window
                outlier_idx(end+1) = handover_idx; % Save outlier idx
                inlier_idx(inlier_idx == handover_idx) = []; % remove outlier idx from inlier list
            end
        end
    
        % % Plot for debug trace_std, trace_mean
        % figure
        % histogram( trace_mean, 10)
        % hold on
        % plot(trace_mean(1,300), 0,'xr')
        % plot([1:length(trace_mean)], trace_mean(1,:,1) ,'xg', [1:length(trace_mean)], trace_mean(1,:,1) + trace_std(1,:,1) ,'xg')
    end
end

% remove duplicates from outlier_idx?
%% build output
handovers_out = handovers(inlier_idx);

end % function end