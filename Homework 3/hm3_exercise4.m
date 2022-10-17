%% Homework 2: Tic tac toe
% Albin Ekstrom
% Date 17 okt 2022


clc
clear variables

% Initilizing parameters
e = 1; % epsilon
epochs = 10000;

% Initilizing q-learning players
player1 = [];
player2 = [];

%% Training

for epoch = 1 : epochs

    % Initilizing Tic tac toe board
    board = zeros([3 3]);
    historic_board = [];
    game_over = false; 

    while game_over
        

        historic_board = [historic_board board];
        
        if true
            game_over = true;
        end

    end

end

%% Functions

% Play tictactoe one round
function tictactoe = play(board)
    
end

% Epsilon Greedy Policy
function idx = e_greedy(epsilon, states)
    
    if rand < epsilon
        idx = 
    else
    end
end
