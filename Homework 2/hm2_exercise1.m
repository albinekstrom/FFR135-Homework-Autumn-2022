%% Homework 2: One-layer perceptron
% Albin Ekström
% Date 27 sep 2022

clc
clear variables
clf

train = readmatrix('CSV/training_set.csv');
val = readmatrix('CSV/validation_set.csv');

% Plot data
ax1 = subplot(1,2,1);
gscatter(ax1,train(:,1),train(:,2),train(:,3))
title(ax1,'Training set')

ax2 = subplot(1,2,2);
gscatter(ax2, val(:,1),val(:,2),val(:,3))
title(ax2,'Validation set')

%% PREPROCESSING INPUT DATA

% Shift and normalizing with variances to 1
for i = 1 : 2
    train(:,i) = (train(:,i) - mean(train(:,i))) / std(train(:,i));
    val(:,i) = (val(:,i) - mean(val(:,i))) / std(val(:,i));
end

% Init parameters
M1 = 15;
max_epochs = 1000;
learning_rate = 0.02;
O_val = zeros(length(val),1);
errors_val = zeros(max_epochs,1);

% Init weight gaussain dist.
w1 = normrnd(0, 1, [M1, 2]); % [10 x 2]
w2 = normrnd(0, 1, [M1, 1]); % [10 x 1]

% Init thresholds as zero vectors
theta1 = zeros(M1, 1); % [10 x 1]
theta2 = zeros(1, 1); % [1]


%% TRAINING & VALIDATION
for epoch = 1 : max_epochs

    for n = 1 : length(train)

        % Pick random input
        nbr = randi(n);
        x = [train(nbr,1); train(nbr,2)];
        t = train(nbr,3);

        % Propagate input throught network
        V = tanh(b(w1, x, theta1)); 
        O = tanh(b(w2', V, theta2)); 

        % Adjust network (GD and backpropagation)
        delta2 = (t - O) .* der_g(b(w2', V, theta2));
        delta1 = (w2 .* delta2) .* der_g(b(w1, x, theta1));
        
        w2 = w2 + learning_rate * delta2 * V; 
        w1 = w1 + learning_rate * delta1 * x';
        
        theta2 = theta2 - learning_rate * delta2;
        theta1 = theta1 - learning_rate * delta1;
    end

    for m = 1 : length(val)

        % Propagate validation set throught network
        x_val = [val(m,1); val(m,2)];
        V_val = tanh(b(w1, x_val, theta1));
        O_val(m) = tanh(b(w2', V_val, theta2));
    end

    % Testing classification error for validation set
    error_val = C(O_val, val(:,3), length(val));
    errors_val(epoch) = error_val;


    if error_val < 0.12
        break;
    end
end

%% PLOT CLASSIFICATION
clf

O_val = sign(O_val);
ax2 = subplot(2,2,1);
gscatter(ax2, val(:,1),val(:,2),O_val)
title(ax2,'Validation Prediction')

test = abs(val(:,3)-O_val);
ax3 = subplot(2,2,2);
gscatter(ax3, val(:,1),val(:,2),test, 'gr')
legend('Correct','Wrong')
title(ax3,'Estimated Predicitions')

ax1 = subplot(2,2,[3,4]);
e = 1:epoch;
plot(ax1, e(10:10:end), errors_val(10:10:epoch), '--or')
title(ax1,'Classification Error for Validation')
ylabel(ax1,'Class. error'), xlabel(ax1,'Epochs')

%% PRINT CSV FILES

csvwrite('CSV/w1.csv',w1)
csvwrite('CSV/w2.csv',w2)

csvwrite('CSV/t1.csv',theta1)
csvwrite('CSV/t2.csv',theta2)

%% FUNCTIONS
function local_field = b(weight, input, threshold)
    local_field = weight * input - threshold;
end

function der_act = der_g(local_field)
    der_act = 1 - tanh(local_field).^2;
end

function c_error = C(O, t, p)
    c_error = 1/(2*p) * sum(abs(sign(O)-t));
end