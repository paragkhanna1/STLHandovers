function [handover_set] = detect_ownership_set(handover_set, OWNERSHIP, debug_plot)
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

thresh = OWNERSHIP.grip_threshold;

% Loop over all sets and handovers
for set_idx = 1:length(handover_set)
    for handover_idx = 1:handover_set(set_idx).N_handovers

        % Get grip signals
        giver_grip = handover_set(set_idx).handover(handover_idx).signals.forces.giver_grip.data';
        taker_grip = handover_set(set_idx).handover(handover_idx).signals.forces.taker_grip.data';

        % Find longest sequence of sum_grip > threshold (pre filter, avoid identifying previous or next handover if recordings overlap)
        [combined, combined_idx] = find_longest_sequence( (giver_grip + taker_grip) > thresh) ;
        
        % Find samples of grip > threshold
        giver_threshed = giver_grip > thresh;
        taker_threshed = taker_grip > thresh;

        % Find longest sequence of grip > threshold
        [giver_owner, giver_owner_idx] = find_longest_sequence((giver_threshed + combined) == 2);
        [taker_owner, taker_owner_idx] = find_longest_sequence((taker_threshed + combined) == 2);
        object_shared = (giver_owner + taker_owner) == 2;
        object_shared_idx = find(object_shared);

        % If detect if object_shared section is empty
        % if max(object_shared) < 1
        %     error("detect_ownership.m: No object shared for set:"+set_idx+" handover:" +handover_idx)
        % end

        % remove overlap from giver and taker owner
        giver_owner = giver_owner - object_shared;
        taker_owner = taker_owner - object_shared;
        
        giver_owner_idx = setdiff(giver_owner_idx, object_shared_idx);
        taker_owner_idx = setdiff(taker_owner_idx, object_shared_idx);

        % Create handover trigger
        handover_trigger = zeros(size(giver_owner));
        handover_trigger_idx = giver_owner_idx(1);
        handover_trigger(handover_trigger_idx) = 1;

        if debug_plot
            % Plot start and end points
            t = handover_set(set_idx).handover(handover_idx).signals.time.data;
            figure(1)
            hold off
            plot(t, giver_grip, 'r', t, taker_grip, 'g')
            hold on
            plot(t(giver_owner_idx),-10*giver_owner(giver_owner_idx), 'xr', t(taker_owner_idx),-15*taker_owner(taker_owner_idx), 'xg')
            grid on
            plot(t(object_shared_idx), -20*object_shared(object_shared_idx), "xb")

            plot(t(handover_trigger_idx), -20, "or")

            legend(["Giver grip", "Taker grip", "Giver owned", "Taker owned", "Shared", "Handover trigger, e"])
            % pause(0.1)

            figure(2)
            
            plot(t(giver_owner_idx(end))-t(giver_owner_idx(1)), 0, "rx");
            plot(t(object_shared_idx(end))-t(object_shared_idx(1)), -10, "gx");
            hold on
        end

        % Save ownership traces in struct
        handover_set(set_idx).handover(handover_idx).signals.ownership.giver_owner.data = giver_owner';
        handover_set(set_idx).handover(handover_idx).signals.ownership.giver_owner.type = "ownership";
        handover_set(set_idx).handover(handover_idx).signals.ownership.taker_owner.data = taker_owner';
        handover_set(set_idx).handover(handover_idx).signals.ownership.taker_owner.type = "ownership";
        handover_set(set_idx).handover(handover_idx).signals.ownership.object_shared.data = object_shared';
        handover_set(set_idx).handover(handover_idx).signals.ownership.object_shared.type = "ownership";

        handover_set(set_idx).handover(handover_idx).signals.ownership.handover_trigger.data = handover_trigger';
        handover_set(set_idx).handover(handover_idx).signals.ownership.handover_trigger.type = "ownership";


    end
end

end

function [longest_sequence, sequence_idx, section_length] = find_longest_sequence(A)
% Find the longest sequence of A(n) == A(n-1)
L=cumsum([1, diff(A)~=0]);
B=splitapply(@sum,A,L);
[section_length, section_in_L] = max(B);
longest_sequence = (L == section_in_L);
sequence_idx = find(longest_sequence);

end