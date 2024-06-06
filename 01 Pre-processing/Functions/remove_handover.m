function [handover_set] = remove_handover(handover_set, set_name, handover_name)
%Removes the handover where handover_set.name == set_name, and
%handover_set.handover.name == handover_name

% Loop over all sets and handovers
for set_idx = 1:length(handover_set)

    if handover_set(set_idx).name == set_name

        to_keep = [1:handover_set(set_idx).N_handovers]; % list all indexes
        for handover_idx = 1:handover_set(set_idx).N_handovers

            if convertCharsToStrings(handover_set(set_idx).handover(handover_idx).name) == convertCharsToStrings(handover_name)
                to_keep = to_keep( to_keep ~= handover_idx ); % Remove specified handover from list
            end

        end
        % Keep all handovers still on the list
        handover_set(set_idx).handover = handover_set(set_idx).handover(to_keep);
        handover_set(set_idx).N_handovers = length(to_keep);

    end
end

