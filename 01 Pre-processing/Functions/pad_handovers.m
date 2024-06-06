function [handover_set] = pad_handovers(handover_set)
%UNTITLED11 Summary of this function goes here

% 

longest_handover = 0;

% Find longest handover ( in samples )
for set_idx = 1:length(handover_set)
    for handover_idx = 1:handover_set(set_idx).N_handovers
        % Get length of handover
        this_len = length( handover_set(set_idx).handover(handover_idx).signals.time.data );
        % Record if length is gt longest so far
        if this_len > longest_handover
            longest_handover = this_len; 
        end
    end
end

Fs = handover_set(set_idx).handover(handover_idx).Fs

% Pad all dimensions to length longest_handover
for set_idx = 1:length(handover_set)
    for handover_idx = 1:handover_set(set_idx).N_handovers

        % Extend time data
        t = handover_set(set_idx).handover(handover_idx).signals.time.data;
        t(end+1 : longest_handover) = t(end) + [1:longest_handover-length(t)]./Fs;
        handover_set(set_idx).handover(handover_idx).signals.time.data = t;

        % Loop over all signals in all frames
        frames = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals)');
        frames = setdiff(frames, "time");

        % Access all frames
        for frame = frames
            % handover_set(set_idx).handover(handover_idx).signals.(frame);

            keys = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals.(frame))');
            
            % Access all signals in this frame
            for key = keys

                    % Add padding to the end of the signal
                    data = handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data;
                    len0 = length(data);
                    for dim = [1:size(data,2)]
                        data(len0+1 : longest_handover, dim) = data(len0, dim);
                    end
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data = data;

            end
        end

    end
end


end % Function end