%% Homework 2: Restricted Boltzmann machine
% Albin Ekström
% Date 30 sep 2022

clc
clear variables

eta = 0.01; % learning rate

M = [1,2,4,8]; % Hidden neurons
N = 3; % Visiable neurons

trails = 100;
minib = 20;
k = 500;

outer = 2000;
inner = 3000;

% Initilize inputs for XOR with prob. 1/4 and else 0
X = [-1	-1	1	1	1	-1	-1	1
     -1	1	-1	1	-1	1	-1	1
     -1	1	1	-1	-1	-1	1	1];

P_D = [1/4 1/4 1/4 1/4 0 0 0 0];
DKL = zeros(1,length(M));

for m = 1 : length(M)
    
    % Initilize random weights and thresholds
    w = normrnd(0, 1, [M(m), N]);
    theta_v = zeros(N, 1); 
    theta_h = zeros(M(m), 1);

    for q = 1 : trails

        % Sample mini batches from XOR in X
        p0 = get_minibatches(X(:,1:4), minib);
        
        % Initilize changes in weights and thresholds
        dw = zeros(M(m), N);
        dt_v = zeros(N, 1);
        dt_h = zeros(M(m), 1);   
        
        for mu = 1 : minib
            v_0 = p0(:,mu);
            
            % Update states of hidden layer
            b_h_0 = b(w, v_0, theta_h);
            h = stochastic(b_h_0);
        
            % Iteration loop
            for t = 1 : k

                % Update states of visiable layer
                b_v = h'*w-theta_v';
                v = stochastic(b_v);
        
                % Update states of hidden layer
                b_h = b(w, v, theta_h);
                h = stochastic(b_h);

            end % iteration loop
               
            % Compute difference in weights and thresholds
            dw = dw + eta*(tanh(b_h_0)*v_0' - tanh(b_h)*v');
            dt_v = dt_v - eta*(v_0-v);
            dt_h = dt_h - eta*(tanh(b_h_0)-tanh(b_h));

        end % mini batches loop
        
        % Update weights and thresholds
        w = w + dw;
        theta_v = theta_v + dt_v;
        theta_h = theta_h + dt_h;

    end % trails
    

    % OUTER LOOP
    P_B = zeros(length(X),1); 
    for o = 1 : outer
        % Select random pattern of 8
        x = X(:,randi(length(X)));
        
        % Set v = xi and update hidden layer
        v = x;
        b_h = b(w, v, theta_h);
        h = stochastic(b_h);

        % INNER LOOP
        for i = 1 : inner
            % Update states of visiable layer
            b_v = h'*w-theta_v';
            v = stochastic(b_v);
    
            % Update states of hidden layer
            b_h = b(w, v, theta_h);
            h = stochastic(b_h);
            
            for mu = 1 : length(X)
                if isequal(X(:,mu), v)
                    P_B(mu) = P_B(mu) + 1/(outer*inner);
                end
            end

        end % inner
    end % outer
    
    % KULLBACK-LEIBLER DIVERGENCE
    
    for p = 1 : 4
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
plot(ax, M_t, DKL_t, 'k-', M, DKL, 'mo')
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

% Generate minibatches
function minibatch = get_minibatches(X, mb)
    minibatch = zeros(3,mb); % batch matrix
    for i = 1:mb
        minibatch(:,i) = X(:,randi(length(X)));
    end
end