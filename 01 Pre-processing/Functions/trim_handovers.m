function [handover_set] = trim_handovers(handover_set, DOWNSAMPLE)
%UNTITLED11 Summary of this function goes here

% Start at first sample of giver_owner = 1
% End at last sample of taker_owner = 1
% Downsample by DOWNSAMPLE.downsample_factor

ds_factor = DOWNSAMPLE.downsample_factor;

% Loop over all sets and handovers
for set_idx = 1:length(handover_set)
    for handover_idx = 1:handover_set(set_idx).N_handovers
        
        % Find start and end index for handover
        Start_idx = find(handover_set(set_idx).handover(handover_idx).signals.ownership.giver_owner.data,1,"first");
        End_idx = find(handover_set(set_idx).handover(handover_idx).signals.ownership.taker_owner.data,1,"last");

        % Set new Fs
        handover_set(set_idx).handover(handover_idx).Fs = handover_set(set_idx).handover(handover_idx).Fs / ds_factor;

        % Set new time scale
        t = handover_set(set_idx).handover(handover_idx).signals.time.data;
        t = downsample( t(Start_idx:End_idx) - t(Start_idx),ds_factor);
        handover_set(set_idx).handover(handover_idx).signals.time.data = t';

        % Loop over all signals in all frames
        frames = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals)');
        frames = setdiff(frames, "time");

        for frame = frames
            % handover_set(set_idx).handover(handover_idx).signals.(frame);

            keys = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals.(frame))');

            for key = keys
                    data = handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data;
                    data = downsample(data(Start_idx:End_idx,:),ds_factor);
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data = data;

            end
        end
    end
end

end % Function end