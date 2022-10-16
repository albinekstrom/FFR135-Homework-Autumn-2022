%% Homework 2: Chaotic time series prediction
% Albin Ekstrom
% Date 15 okt 2022

clc
clear variables
clf

% Read CSV files
train = readmatrix('CSV/training-set.csv');
X_test = readmatrix('CSV/test-set-10.csv');

% Preprocessing
X = train(:,1:end-1);
Y = train(:,2:end);

% Plot data 3D
plot3(train(1,:),train(2,:),train(3,:),'r')

%% Initializing

% Reservoir
N = 3; % inputs
R = 500; % reservoir neurons
O = 3; % outputs

k = 0.01; % ridge para.
T = 500;

w_in = normrnd(0,sqrt(0.002),[R,N]);
W = normrnd(0,sqrt(0.004),[R,R]);

%% Training Reservoir

% Creating reservoir hidden neurons
r = zeros([R,length(train)]);

for t = 1 : length(train)-1
    r(:,t+1) = tanh(W * r(:,t) + w_in * X(:,t));
end

% Remove inital columns in reservoir
r = r(:,51:end);
Y = Y(:,50:end);

% Update w_out matrix
w_out = Y*r'*inv(r*r'+k*eye(R));

%% Predicting

% Creating reservoir hidden neurons
r_p = zeros(R,1);

% Initialize reservoir
for t = 1 : length(X_test)
    r_p = tanh(W * r_p + w_in * X_test(:,t));
end

output = zeros(O,T);

% Determine output
for t = 1 : T
    output(:,t) = w_out * r_p;
    r_p = tanh(W * r_p + w_in * output(:,t));
end

%% Plot output data

% Plot data 3D
plot3(X_test(1,:),X_test(2,:),X_test(3,:),'b'), hold on
plot3(output(1,:),output(2,:),output(3,:),'r')

%% Output data to CSV

csvwrite('CSV/prediction.csv',output(2,:))