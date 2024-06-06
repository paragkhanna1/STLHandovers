function [handover_set] = trim_handovers_pickup_zone(handover_set, DOWNSAMPLE)
%UNTITLED11 Summary of this function goes here

% Start at leaving_pickup_zone
% End at last sample of taker_owner = 1
% Downsample by DOWNSAMPLE.downsample_factor

ds_factor = DOWNSAMPLE.downsample_factor;

% Loop over all sets and handovers
for set_idx = 1:length(handover_set)
    for handover_idx = 1:handover_set(set_idx).N_handovers
        
        % Find start and end index for handover
        Start_idx = handover_set(set_idx).handover(handover_idx).values.leaving_pickup_zone;
        End_idx = find(handover_set(set_idx).handover(handover_idx).signals.ownership.object_shared.data,1,"last");
        if isempty(End_idx)
            End_idx = find(handover_set(set_idx).handover(handover_idx).signals.ownership.giver_owner.data,1,"last");
        end

        if Start_idx >= End_idx
            
            error("trim_handovers_pickup_zone.m: Start index >= End index for set:"+set_idx+" handover:"+handover_idx)
        end

        % Set new Fs
        handover_set(set_idx).handover(handover_idx).Fs = handover_set(set_idx).handover(handover_idx).Fs / ds_factor;

        % Set new time scale
        t = handover_set(set_idx).handover(handover_idx).signals.time.data;
        if isempty(t(Start_idx:End_idx) - t(Start_idx))
            foo = "what?"
        end
        t = downsample( t(Start_idx:End_idx) - t(Start_idx), ds_factor);
        handover_set(set_idx).handover(handover_idx).signals.time.data = t';

        % Loop over all signals in all frames
        frames = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals)');
        frames = setdiff(frames, "time");

        for frame = frames
            % handover_set(set_idx).handover(handover_idx).signals.(frame);

            keys = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals.(frame))');

            for key = keys
                    data = handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data;
                    data = downsample(data(Start_idx:End_idx,:),ds_factor); % Trim and downsample
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data = data;

            end
        end
    end
end

end % Function end