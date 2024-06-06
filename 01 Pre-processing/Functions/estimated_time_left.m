function [string , time_left] = estimated_time_left(loop_time,loops_left)
%[string , time_left] = estimated_time_left(loop_time,loops_left)
%   Detailed explanation goes here

time_left = loops_left * loop_time;
s_left = mod(time_left, 60);
min_left = (time_left - s_left)/60;
string = ("Cycles left = " + loops_left + "; Cycle time = "+ loop_time+"s \nPredicted time left "+min_left+"min, "+s_left+"s.\n");
end