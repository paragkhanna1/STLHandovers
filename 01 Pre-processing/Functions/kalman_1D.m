function [pos, vel] = kalman_1D(measured_position, Fs, varargin)
% [possition, velocity] = kalman_1D(measured_position, alpha)
%   Input
%       measured_position   Nx1 Noisy position signal
%       alpha               1x1 R1/R2 for Kalman filter, 10 by default
%   Output
%       pos                 Nx1 Estimated position
%       vel                 Nx1 Estimated velocity

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

% x = [pos, vel]'
x_hat = zeros(2, N+1);
x_hat(:,1) = [y(1), 0]';
y_hat = zeros(1, N+1);

% alpha = 10; % alpha = R1/R2 % set in input parser
% High alpha
%   - Track measurements
% Low alpha
%   - Reject noise
R1 = 1;
R2 = R1/alpha;
Q = eye(2);

% Model
F = [1, 1/Fs;
    0, 1];
G = [0, 1]'; % Random walk velocity
H = [1, 0];

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

end