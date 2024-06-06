function [handover_set] = write_dataset(handover_set, OUTPUT_SIGNALS, OUTPUT_FILEPATH)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here

mkdir(OUTPUT_FILEPATH.folder);

% Loop over all sets and handovers
for set_idx = 1:length(handover_set)
    for handover_idx = 1:handover_set(set_idx).N_handovers
        % Define file name
        filename = handover_set(set_idx).name + "_" + handover_set(set_idx).handover(handover_idx).name;
        filename = replace(filename, ["/", " "],"_");

        % Get signals
        t = handover_set(set_idx).handover(handover_idx).signals.time.data;
        t_len = length(t);

        % Build table
        Table_array = [];
        Table_array(1:t_len, 1) = t;
        Table_names = string(["time"]);
        column_idx = 1;

        % Loop through all output signals
        for signal = OUTPUT_SIGNALS
            column_idx = column_idx +1;
            frame = signal.frame;
            key = signal.trace;
            if signal.type == "Velocity"
                key = key + "_vel";
            end
            if signal.type == "Acceleration"
                key = key + "_acc";
            end

            data = handover_set(set_idx).handover(handover_idx).signals.(signal.frame).(key).data(:,signal.axis);
            Table_array(:,column_idx) = data;
            Table_names(column_idx) = signal.name;


        end
        T = array2table(Table_array, VariableNames=Table_names);
        handover_set(set_idx).handover(handover_idx).table = T;

        
        writetable(T, OUTPUT_FILEPATH.folder + filename + ".csv");
    end
end

end % function end