%% Homework 2: Restricted Boltzmann machine
% Albin Ekström
% Date 30 sep 2022

clc
clear variables

eta = 0.01; % learning rate

M = [1,2,4,8]; % Hidden neurons
N = 3; % Visiable neurons

trails = 1000;
minibatches = 20;
k = 2000;

outer = 2000;
inner = 3000;

% Initilize inputs for XOR with prob. 1/4 and else 0
X = [-1	-1	1	1	1	-1	-1	1
     -1	1	-1	1	-1	1	-1	1
     -1	1	1	-1	-1	-1	1	1];


P_B = zeros(length(X),1); 
P_D = [1/4 1/4 1/4 1/4 0 0 0 0];
DKL = zeros(length(M),1);

for m = length(M)
    
    % Initilize random weights and thresholds
    w = normrnd(0, 1, [M(m), N]);

    theta_v = zeros(N, 1); 
    theta_h = zeros(M(m), 1);

    for t = 1 : trails
        mini_batch = 

        % Generate random pattern from input
        r = randi(4);
        v_0 = X(:,r);
        
        % Update states of hidden layer
        b_h_0 = b(w, v_0, theta_h);
        h = stochastic(b_h_0);
    
        % INNER LOOP
        for i = 1 : iteration
            % Update states of visiable layer
            b_v = h'*w-theta_v';
            v = stochastic(b_v);
    
            % Update states of hidden layer
            b_h = b(w, v, theta_h);
            h = stochastic(b_h);
    
            if isequal(v, v_0)
                P_B(r) = P_B(r) + 1/(outer*inner);
            end
    
        end
    
        % Update weights and thresholds
        w = w + eta*(tanh(b_h_0).*v_0' - tanh(b_h).*v');
        theta_v = theta_v - eta*(v_0-v);
        theta_h = theta_h - eta*(tanh(b_h_0)-tanh(b_h));
        
    end
    
    % KULLBACK-LEIBLER DIVERGENCE
    
    for p = 1 : length(M)
        if P_B(p) == 0; lgPB = 0; 
        else; lgPB = log(P_B(p)); end
    
        DKL(m) = DKL(m) + P_D(p)*(log(P_D(p))-lgPB);
    end
end

%% THEORETICAL VALUE OF DKL
M_t = 1:9;
DKL_t = zeros(length(M_t),1);

for i = 1 : length(M_t)
    if M_t(i) < 2^(N-1)-1
        DKL_t(i) = N - (log2(M_t(i)+1)) - (M_t(i)+1)/(2^(log2(M_t(i)+1)));
    else
        DKL_t(i) = 0;
    end
end

%% PLOT DKL GRAPH
clc

ax = gca;
plot(ax, M_t, DKL_t)
title('Kullback-Leiber divergence theoretical vs approx.')
ylabel(ax,'Kullback-Leiber divergence [D_{KL}]'), xlabel(ax,'Hidden nerons [M¨]')
axis([0.5 9 -0.05 1.1])

%% FUNCTIONS
% Local field
function local_field = b(weigth, input, threshold)
    local_field = weigth * input - threshold;
end

% Stochastic update
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