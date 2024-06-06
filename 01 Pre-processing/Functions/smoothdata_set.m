function [handover_set] = smoothdata_set(handover_set)
%Smooths data and estimates speed using Kalman filter on every signal in
%every frame
%   Detailed explanation goes here

for set_idx = 1:length(handover_set)
    for handover_idx = 1:handover_set(set_idx).N_handovers
        frames = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals)');
        frames = setdiff(frames, "time");

        Fs = handover_set(set_idx).handover(handover_idx).Fs;

        for frame = frames
            handover_set(set_idx).handover(handover_idx).signals.(frame);

            keys = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals.(frame))');
            keys = unique(erase(keys, "_orientation"));

            for key = keys
                    pos = handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data;
                    vel = zeros(size(pos));

                    alpha = 10;

                    for dim = 1:size(pos,2)
                        [pos(:,dim), vel(:,dim)] = kalman_1D(pos(:,dim), Fs, "alpha", alpha);
                    end
                
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data = pos;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key).kalman_alpha = alpha;

                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").data = vel;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").type = "velocity";
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").name = key + "_vel";
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").trace = key;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").kalman_alpha = alpha;
            end

        end
    end
end

end