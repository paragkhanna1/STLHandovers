function [handover_set] = detect_pickup_zone(handover_set, pickup_zone_radius)
%UNTITLED Summary of this function goes here
%   Detects pick up zone and "in pickup zone" propperty of giver

%   pickup zone for a handover signal the position of the hand at the first
%   index of giver_ownership.data == 1

% pickup_zone_radius = 0.1

% Loop over all sets and handovers
for set_idx = 1:length(handover_set)
    for handover_idx = 1:handover_set(set_idx).N_handovers
        
        handover_set(set_idx).handover(handover_idx).values.pickup_zone_radius = pickup_zone_radius;

        % Get start of 1st giver_ownership
        t0 = find( handover_set(set_idx).handover(handover_idx).signals.ownership.giver_owner.data == 1, 1, "first" );

        % Get hand position at t0
        pickup_location = handover_set(set_idx).handover(handover_idx).signals.map.giver_RHand.data(t0,:);

        % Save pickup zone in handover
        handover_set(set_idx).handover(handover_idx).values.pickup_location = pickup_location;

        % find where giver_RHand is in the pickup zone
        dist_from_pickup_location = sqrt(sum((handover_set(set_idx).handover(handover_idx).signals.map.giver_RHand.data(:,:) - pickup_location).^2, 2));
        in_pickup_zone = (dist_from_pickup_location <= pickup_zone_radius);

        handover_set(set_idx).handover(handover_idx).signals.pickup_zone.dist_from_pickup_location.data = dist_from_pickup_location;
        handover_set(set_idx).handover(handover_idx).signals.pickup_zone.dist_from_pickup_location.type = 'misc';
        handover_set(set_idx).handover(handover_idx).signals.pickup_zone.in_pickup_zone.data = in_pickup_zone;
        handover_set(set_idx).handover(handover_idx).signals.pickup_zone.in_pickup_zone.type = 'misc;'

        % Find first 0 index of in_pickup_zone after t0
        % when giver RHand leaves handover zone, finds 1st index outside pickup_zone
        leaving_pickup_zone = t0 + find( in_pickup_zone(t0:end) == 0, 1, "first") - 1;

        handover_set(set_idx).handover(handover_idx).values.leaving_pickup_zone = leaving_pickup_zone;

        % plot for science
        % t = handover_set(set_idx).handover(handover_idx).signals.time.data;
        % Gx_map = handover_set(set_idx).handover(handover_idx).signals.map.giver_RHand.data(:,1);
        % plot(t, Gx_map, t(t0), pickup_location(1), 'x', t(in_pickup_zone), Gx_map(in_pickup_zone), 'r', t(leaving_pickup_zone), Gx_map(leaving_pickup_zone), 'gx')
        % figure(2)
        % pause(0.6)
    end
end

end