function [pos, vel, acc] = kalman_1D_acc(measured_position, Fs, varargin)
% [possition, velocity, acceleration] = kalman_1D(measured_position, alpha)
%   Input
%       measured_position   Nx1 Noisy position signal
%       alpha               1x1 R1/R2 for Kalman filter, 10 by default
%   Output
%       pos                 Nx1 Estimated position
%       vel                 Nx1 Estimated velocity
%       acc                 Nx1 Estimated acceleration

% parse inputs
p = inputParser;
addParameter(p, "alpha", 10, @isnumeric)
parse(p,varargin{:})

alpha = p.Results.alpha;

N = length(measured_position);

%% Kalman filter

% Measurement
% y = [pos]
y = zeros(1,N+1);
y(2:end) = measured_position;
y(1) = y(2);

% x = [pos, vel, acc]'
x_hat = zeros(3, N+1);
x_hat(:,1) = [y(1), 0, 0]';
y_hat = zeros(1, N+1);

% alpha = 10; % alpha = R1/R2 % set in input parser
% High alpha
%   - Track measurements
% Low alpha
%   - Reject noise
R1 = 1;
R2 = R1/alpha;
Q = eye(3); % Measurement noise (estimated)

% Model
F = [1,     1/Fs,   1/(2*Fs^2)  ;
     0,     1,      1/Fs        ;
     0,     0,      1           ]; % State update matrix
G = [0, 0, 1]'; % Random walk acceleration
H = [1, 0, 0];

% Filter loop
for n = 1:801
    % Predict:
    x_hat(:,n+1) = F * x_hat(:,n);
    y_hat(:,n+1) = H * x_hat(:,n+1);
    P = F*Q*F' + G*R1*G';

    % Update:
    L = (P*H')/(H*P*H' + R2);
    x_hat(:,n+1) = x_hat(:,n+1) + L*(y(n+1) - H*x_hat(:,n+1));
    Q = P - P*H'/(H*P*H' + R2) * H*P;
end

% cut the init
x_hat = x_hat(:,2:end);
% y_hat = y_hat(:,2:end);

pos = x_hat(1,:)';
vel = x_hat(2,:)';
acc = x_hat(3,:)';
end