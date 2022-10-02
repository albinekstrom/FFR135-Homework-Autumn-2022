%% Homework 2: Restricted Boltzmann machine
% Albin Ekström
% Date 30 sep 2022

clc
clear variables
clf

eta = 0.01; % learning rate

M = 4; % Hidden neurons
N = 3; % Visiable neurons

outer = 2000;
inner = 3000;

% Initilize inputs for XOR with prob. 1/4 and else 0
X = [-1	-1	1	1	1	-1	-1	1
     -1	1	-1	1	-1	1	-1	1
     -1	1	1	-1	-1	-1	1	1];
P_D = [1/4 1/4 1/4 1/4 0 0 0 0];

% Initilize random weights and thresholds
w = normrnd(0, 1, [M, N]);

theta_v = zeros(N, 1); 
theta_h = zeros(M, 1);

P_B = zeros(length(X),1);

% Outer loop
for o = 1 : 1
    % Generate random pattern from input
    v_0 = X(:,randi(length(X)));
    
    % Update states of hidden layer
    b_h_0 = b(w, v_0, theta_h);
    h = stochastic(b_h_0);

    % Inner loop
    for i = 1 : 1
        % Update states of visiable layer
        b_v = h'*w-theta_v';
        v = stochastic(b_v);

        % Update states of hidden layer
        b_h = b(w, v, theta_h);
        h = stochastic(b_h);

        for j = 1 : length(X)
            disp(v)
            disp(X(:,j))
            disp(isequal(v, X(:,j)))
            if isequal(v, X(:,j))
               P_B(j) = P_B(j) + 1/(outer*inner);
            end
        end

    end % inner

    % Update weights and thresholds
    w = w + eta*(tanh(b_h_0).*v_0' - tanh(b_h).*v');
    theta_v = theta_v - eta*(v_0-v);
    theta_h = theta_h - eta*(tanh(b_h_0)-tanh(b_h));

end % outer

%% KULLBACK-LEIBLER DIVERGENCE



%% FUNCTIONS
function local_field = b(weigth, input, threshold)
    local_field = weigth * input - threshold;
end

function pm = stochastic(local_field)
    pm = zeros(length(local_field),1);
    for i = 1 : length(local_field)
        prob = 1/(1+exp(-2*local_field(i)));
        r = randn(1);
        if r < prob
            pm(i) = 1;
        else
            pm(i) = -1;
        end
    end
end