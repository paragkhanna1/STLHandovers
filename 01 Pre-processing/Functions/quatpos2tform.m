function [TF] = quatpos2tform(quat,pos)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

TF = quat2tform(quat);

TF(1:3,4,:) = pos';
end