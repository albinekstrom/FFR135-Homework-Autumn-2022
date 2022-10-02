clc
clear all

steps = 10^4;
epochs = 20;

dim = [2,3,4,5];
eta = 0.05;
counter = [];
syms k;

for n = dim
    % Create INPUT matrix (n x 2^n)
    boolN = symsum(2^(k),k,0,n-1);
    boolInput = (dec2bin(0:sum(boolN)) - '0')';

    % Generate dummy matrix to use ismember
    boolUsed = zeros(2^2^n, 2^n);
    count = 0;
    for s = 1 : steps
        % Randomize TARGET matrix (1 x 2^n) with only [-1, 1]
        boolOutput = 2 * randi([0, 1], [1,2^n]) - 1;

        if ~ismember(boolOutput, boolUsed, 'rows')
           % Randomize weight matrix with std 1/sqrt(n) and mean 0
           w = randn(1,n)*1/sqrt(n); 
           phi = 0;
           for e = 1 : epochs
              totalError = 0;
              for j = 1 : (2^n)
                  % OUTPUT matrix
                  O = sign(sum(w * boolInput(:,j)) - phi);
                  error = boolOutput(:,j) - O;
                  
                  % Update w and phi
                  w = w + eta * (boolOutput(j) - O) * boolInput(:,j)'; 
                  phi = phi + (-eta) * (boolOutput(j) - O);
                  totalError = totalError + abs(error);
              end
              if totalError == 0
                  count = count + 1;
                  break
              end
           end
           boolUsed(s,:) = boolOutput;
        end
    end
    counter = [counter count];
end