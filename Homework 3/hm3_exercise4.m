%% Homework 2: Tic-Tac-Toe
% Albin Ekstrom
% Date 20 okt 2022

clc
clear variables
clf

% Initilizing parameters
lr = 0.1;
games = 50000;
step = 10;

% Initilizing q-learning players
p1 = dictionary();
p2 = dictionary();


%% Training

epsilon = 1;

p1_win = 0;
p2_win = 0;
draws = 0;
count_wins = [];

for game = 1 : games
    
    % Initilizing Tic Tac Toe
    board = zeros([3 3]);

    states1 = [];
    historic1 = [];

    states2 = [];
    historic2 = [];

    while true
        % player 1 moves
        historic1 = [historic1; board];
        [p1, board, state1] = move(p1, 1, board, epsilon); 
        states1 = [states1; state1];

        [reward, gg] = game_over(board);
        if gg
            break;
        end
        
        % player 2 moves
        historic2 = [historic2; board];
        [p2, board, state2] = move(p2, -1, board, epsilon); 
        states2 = [states2; state2];

        [reward, gg] = game_over(board);
        if gg
            break;
        end
    end

    % Give reward to each player
    p1 = feed_reward(p1, states1, historic1, reward, lr);
    p2 = feed_reward(p2, states2, historic2, -reward, lr);
    
    % Update win steaks
    if reward == 1
        p1_win = p1_win + 1;
    elseif reward == -1
        p2_win = p2_win + 1;
    else
        draws = draws + 1;
    end

    if ~(mod(game,step))
        epsilon = epsilon * 0.99;
        wins = [p1_win p2_win draws]/game;
        count_wins = [count_wins; wins];
    end
    
end

%% Plot wins
clc
x = 1:step:games;

plot(x,count_wins(:,1),'b'), hold on
plot(x,count_wins(:,2),'r')
plot(x,count_wins(:,3),'g')
legend("Player 1 Wins", "Player 2 Wins", "Draw")
xlabel('Number of Games')
ylabel('Fraction of Games Won/Drawn')

%% Play a game

p1_win = 0;
p2_win = 0;
draws = 0;

for i = 1 : 20000

    board = zeros([3 3]);
    
    historic1 = [];
    historic2 = [];
    
    while true
        historic1 = [historic1; board];
        [p1, board, state1] = move(p1, 1, board, epsilon); % player 1 moves
        
        [reward, gg] = game_over(board);
        if gg
            [~, p1] = get_q_table(p1, board);
            [~, p2] = get_q_table(p2, board);
            historic2 = [historic2; board];
            
            break;
        end
        
        historic2 = [historic2; board];
        [p2, board, state2] = move(p2, -1, board, epsilon); % player 2 moves
        
        [reward, gg] = game_over(board);
        if gg
            [~, p1] = get_q_table(p1, board);
            [~, p2] = get_q_table(p2, board);
            historic1 = [historic1; board];
            
            break;
        end
    end
    
    if reward == 1
        p1_win = p1_win + 1;
    elseif reward == -1
        p2_win = p2_win + 1;
    else
        draws = draws + 1;
    end

end

%% Ouput CSV
QP1 = zeros([6 3*length(p1)]);
QP2 = zeros([6 3*length(p2)]);

keysP1 = keys(p1);
keysP2 = keys(p2);

for l = 1 : length(keysP1)
    % Get the board and assign it to boards
    strBoard = keysP1(l);
    board = str2num(strBoard);
    QP1(1:3,3*l-2:3*l) = board;

    q_cell = p1(strBoard);
    q_table = cell2mat(q_cell);
    QP1(4:6,3*l-2:3*l) = q_table;
end

for l = 1 : length(keysP2)
    % Get the board and assign it to boards
    strBoard = keysP2(l);
    board = str2num(strBoard);
    QP2(1:3,3*l-2:3*l) = board;

    q_cell = p2(strBoard);
    q_table = cell2mat(q_cell);
    QP2(4:6,3*l-2:3*l) = q_table;
end

csvwrite('CSV/player1.csv',QP1)
csvwrite('CSV/player2.csv',QP2)


%% Functions

% Make one move for specific player
function [player, board, state] = move(player, p_type, board, epsilon)

    % Save or get state
    [q_table, player] = get_q_table(player, board);
    
    % Type of move
    if rand < epsilon
        state = random_move(board);
    else
        state = max_state(q_table);
    end
    
    % Make a move
    board(state(1), state(2)) = p_type;

end


% Save board state or get q-table from board state
function [q_table, player] = get_q_table(player, board)
    
    % Convert matrix to string
    strBoard = mat2str(board);

    % Try to find strBoard in Q for player
    try 
        % Return q_value for given board
        q_cell = player(strBoard);
        q_table = cell2mat(q_cell);

    catch
        q_table = board;
        q_table(board == 1 | board == -1) = NaN;
        player(strBoard) = {q_table};
    end
end


% Make random move
function state = random_move(board)

    % Finding available states on board
    [row, col] = find(board == 0);
    
    % Picking random available next state
    nbr = randi(length(row));
    state = [row(nbr) col(nbr)];

end


% Finding max value from Q
function state = max_state(q_value)  

    maximum = max(max(q_value));
    [row, col] = find(q_value==maximum);

    if length(row) > 1
        nbr = randi(length(row));
        state = [row(nbr) col(nbr)];
    else
        state = [row col];
    end

end


% Check if game over
function [winner, gg] = game_over(board)

    sum_cols = sum(board);
    sum_rows = sum(board,2);
    sum_diag = trace(board);
    sum_diag_T = sum(diag(board(:,end:-1:1)));

    sums = [sum_cols sum_rows' sum_diag sum_diag_T];

    if find(sums == 3)
        winner = 1;
        gg = true;
    elseif find(sums == -3)
        winner = -1;
        gg = true;
    elseif nnz(board) == 9
        winner = 0;
        gg = true;
    else
        winner = 0;
        gg = false;
    end

end


function player = feed_reward(player, states, boards, reward, lr)

    % Count boards
    b_count = (length(boards)/3); 

    % Backpropagate for all other boards
    for i = b_count : -1.0 : 1.0
        board = boards(3*i-2:3*i,:);
        state = states(i,:);
        
        % Convert matrix to string
        strBoard = mat2str(board);
        q_table = cell2mat(player(strBoard));
        
        % Give reward for move
        dQ = lr * (reward - q_table(state(1), state(2)));
        q_table(state(1), state(2)) = q_table(state(1), state(2)) + dQ;

        % Update Q-table board
        player(strBoard) = {q_table};

        % Update reward
        reward = q_table(state(1), state(2));
    end

end


