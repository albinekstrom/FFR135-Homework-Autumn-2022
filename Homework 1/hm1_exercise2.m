%% One-step error probability

clc
clear all

patterns = [12; 24; 48];
N = 120;
indp_trails = 10^5;
W = zeros(N,N);
error_probs = [];

for k = 1:length(patterns)
    p = patterns(k);
    errors = 0;
    
    for i = 1 : indp_trails
        x_rand = randi(p);
        n_rand = randi(N);

        X = 2 * randi([0, 1], [N,p]) - 1;

        W = 1/N * X(n_rand,:) * X';
        W(n_rand) = 0;
        
        Bn = W * X(:,x_rand);
        if sign(Bn) ~= 0; S_t = sign(Bn); else 
            S_t = 1; end
        
        if S_t~=X(n_rand, x_rand)
            errors = errors + 1;
        end
    end

    P = errors/indp_trails;
    error_probs(end+1) = P;
end

 