function [handover_set] = extimate_pos_vel_acc(handover_set, threshold, alpha)
% Smooths data and estimates speed and acceleration using Kalman filter as
%   state estimator on every signal in every frame
%   And removes handovers with significant tracking errors, if measurement
%   uncertainty exceedes a threshold

%   Detailed explanation goes here

% Record error in tracking
max_tracking_error = [];
mean_abs_tracking_error = [[],[],[]];

for set_idx = 1:length(handover_set)
    
    % Record handovers with unacceptable tracking error
    outlier_handovers = [];

    for handover_idx = 1:handover_set(set_idx).N_handovers

        frames = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals)');
        frames = setdiff(frames, ["time", "forces", "ownership", "pickup_zone"]);

        Fs = handover_set(set_idx).handover(handover_idx).Fs;

        for frame = frames
            handover_set(set_idx).handover(handover_idx).signals.(frame);

            keys = string(fieldnames(handover_set(set_idx).handover(handover_idx).signals.(frame))');
            keys = unique(erase(keys, "_orientation"));

            for key = keys
                    pos = handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data;
                    pos0 = pos; % Save position vector signal before filtering, to 
                    vel = zeros(size(pos));
                    acc = zeros(size(pos));
                    tracking_error = zeros(size(pos));

                    % alpha = 50;

                    for dim = 1:size(pos,2)
                        [pos(:,dim), vel(:,dim), acc(:,dim)] = kalman_1D_acc(pos(:,dim), Fs, "alpha", alpha);
                        
                        % Get estimated tracking error (x_hat - x_measured)
                        tracking_error(:,dim) = pos(:,dim) - handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data(:,dim);

                        % % plot for inspection/ verification/ SCIENCE!
                        % t = handover_set(set_idx).handover(handover_idx).signals.time.data;
                        % figure
                        % plot(t, handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data(:,dim))
                        % hold on
                        % plot(t, pos(:,dim))
                        % hold on
                        % plot(t, vel(:,dim))
                        % hold on
                        % plot(t, acc(:,dim))
                        % legend({'raw pos', 'pos', 'vel', 'acc'})
                        % hold off
                        % title("Set: "+set_idx+"; Handover: "+handover_idx+" ; signal: " +key + dim)
                    end

                    % Record error in tracking
                    max_tracking_error(end+1) = max(abs(tracking_error), [],"all");
                    mean_abs_tracking_error(end+1, 1:3) = mean(abs(tracking_error));

                    % Test error threshold
                    % if max_tracking_error(end) > threshold %&& max_tracking_error(end) > threshold +0.12
                    %     axis_label = ["x","y","z"];
                    %     figure
                    %     sgtitle( replace(handover_set(set_idx).name, "_", " ") + " " + replace(handover_set(set_idx).handover(handover_idx).name, "_", " ") + " " + replace(key, "_", " ") + "; tracking error: " + max_tracking_error(end))
                    %     for i = [1,2,3]
                    %         subplot(3,1,i)
                    %         plot(handover_set(set_idx).handover(handover_idx).signals.time.data, handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data(:,i))
                    %         ylabel(axis_label(i) + "[m]")
                    %         xlabel("time [s]")
                    %     end
                    % 
                    %     figure
                    %     plot(handover_set(set_idx).handover(handover_idx).signals.time.data , tracking_error)
                    %     sgtitle( replace(handover_set(set_idx).name, "_", " ") + " " + replace(handover_set(set_idx).handover(handover_idx).name, "_", " ") + " " + replace(key, "_", " ") + "; tracking error: " + max_tracking_error(end) )
                    %     legend({"x est - x measured","y est - y measured","z est - z measured"})
                    %     xlabel("time [s]")
                    %     ylabel("Tracking error [m]")
                    %     pause(0.1)
                    % end


                    % If tracking error > threshold record as outlier
                    if max_tracking_error(end) > threshold
                        outlier_handovers(end+1) = handover_idx;
                    end
                
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key).data = pos;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key).kalman_alpha = alpha;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key).tracking_error = tracking_error;

                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").data = vel;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").type = "velocity";
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").name = key + "_vel";
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").trace = key;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_vel").kalman_alpha = alpha;

                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_acc").data = acc;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_acc").type = "acceleration";
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_acc").name = key + "_acc";
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_acc").trace = key;
                    handover_set(set_idx).handover(handover_idx).signals.(frame).(key + "_acc").kalman_alpha = alpha;
            end

        end
    end

    % Remove all outlier handovers
    inliers = setdiff([1:handover_set(set_idx).N_handovers], outlier_handovers)
    handover_set(set_idx).handover = handover_set(set_idx).handover(inliers)
    handover_set(set_idx).N_handovers = length(inliers)
end

% % Plot histogram of max tracking error
% figure
% histogram(max_tracking_error)
% title("Max estimation error distribution")
% ylabel('frequency')
% xlabel('Max estimation error [m]')
% 
% % Plot histogram of mean abs tracking error
% figure
% axis_label = ['x' 'y' 'z'];
% for i = [1,2,3]
%     subplot(1,3,i)
%     histogram(mean_abs_tracking_error(:,i))
%     ylabel(axis_label(i))
% end
% sgtitle("Tracking error mean abs distribution")
% asdf=0;
end