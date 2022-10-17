%% Homework 2: Self organising map
% Albin Ekstrom
% Date 16 okt 2022

clc
clear variables

% Read CSV files
X = readmatrix('CSV/iris-data.csv');
t = readmatrix('CSV/iris-labels.csv');

%% Initilizing

w_shape = [40,40,4];
output_shape = [40,40];

% Learning rate
eta_0 =  0.1; % initial learning rate
eta_d = 0.01; % decay rate

% Neighbourhood function
sigma_0 = 10; % initial value
sigma_d = 0.05; % decay rate

% Standardise input data
X = X./max(X);

% Initializing train parameters
epochs = 10;
p = length(t); % data length stochastic updates

W_0 = rand(w_shape);
D = zeros(output_shape); % distance matrix

%% Training

% Initilizing weigths
W = W_0;

% Start training
for e = 1 : epochs

    % Update learning rate and neighbourhood function
    eta = eta_0*exp(-eta_d * e);
    sigma = sigma_0*exp(-sigma_d * e);

    for data_point = 1 : p
        
        % Choose a data point randomly
        nbr = randi(p);
        xr = X(nbr,:);

        [i,j] = get_min_distance(W, xr, D, w_shape);
    
        % Update weight matrix
        if D(i,j) < 3 / sigma

            idx0 = [i,j]; % min distance
            dw = zeros(w_shape); % empty delta matrix

            for i = 1 : w_shape(1)
                for j = 1 : w_shape(2)
                    Q = eta * h([i,j], idx0, sigma);
                    for k = 1 : w_shape(3)

                    % Delta weight matrix
                    dw(i,j,k) = Q * (xr(k) - W(i,j,k));
                    end
                end
            end

            W = W + dw; % update
        end

    end % data length

end % epochs

%% Predicitons trained weigths 

iris_rand = zeros([40 2]);
iris = zeros([40 2]);

for x = 1 : p

    % Predicting winning neuron for each flower in data
    [i0,j0] = get_min_distance(W_0,X(x,:),D,w_shape); % random weights
    [i,j] = get_min_distance(W,X(x,:),D,w_shape); % trained weights 
    
    iris_rand(x,:) = [i0,j0];
    iris(x,:) = [i,j];
end

%% Plot data
clc
ax1 = subplot(1,2,1);
gscatter(iris_rand(:,1), iris_rand(:,2), t)
title(ax1,'Random Initial Weights')
legend('Iris Setosa','Iris Versicolour','Iris Virginica');

ax2 = subplot(1,2,2);
gscatter(iris(:,1), iris(:,2), t)
title(ax2,'After Iterating the Learning Rule')
legend(ax2,'off')

linkaxes([ax1 ax2]);
ax1.XLim = [-5 45];
ax2.YLim = [-5 45];


%% Functions

function [row,col] = get_min_distance(weight, x, dist_matrix, shape)
    for i = 1 : shape(1) % 40
        for j = 1 : shape(2) % 40
            d = 0; % distance for i,j to i0,j0
            for k = 1 : shape(3) % 4
                d = d + (weight(i,j,k) - x(k))^2;
            end
            dist_matrix(i,j) = sqrt(d);
        end
    end

    % Fidning minimum value in D
    m = min(min(dist_matrix));
    [row,col] = find(dist_matrix == m);
end

function neigh_func = h(i, i0, s)
    neigh_func = exp(-1/(2*s^2) * sum((i - i0).^2));
end