function [handover_set_out] = transform_set(handover_set, FRAMES, all_frames, all_trace_keys)
%[handover_set_out] = transform_set(handover_set, FRAMES, all_frames, all_trace_keys)
%   Transform all positions and orientations of 'all_trace_keys' from map 
%   frame to 'all_frames', as specified in "FRAMES".

% Keep only position trace keys
trace_keys = string([]);
for key = all_trace_keys
    try
        if handover_set(1).handover(1).signals.map.(key).type == "position"
            trace_keys(end+1) = key;
        end
    catch ME
        if not (ME.identifier == 'MATLAB:nonExistentField')
            rethrow(ME)
        end
    end
end

% loop over all sets
for set_idx = 1:length(handover_set)
    
    fprintf("\nTransforming %d handovers from " + handover_set(set_idx).name + "\n", handover_set(set_idx).N_handovers)

    % loop over all handovers
    for handover_idx = 1:handover_set(set_idx).N_handovers

        for frame = all_frames
            p1 = FRAMES.(frame).point1;
            p2 = FRAMES.(frame).point2;
        
            handover_set(set_idx).handover(handover_idx).signals.(frame) = ...
                p2p_Basili_frame(handover_set(set_idx).handover(handover_idx).signals.map,...
                p1, p2, "Trace_keys", trace_keys);
           
        end
        
    end

    
end

handover_set_out = handover_set;

end % function end