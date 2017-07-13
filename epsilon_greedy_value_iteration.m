function [ policy ] = epsilon_greedy_value_iteration( P, R, discount, epsilon )
%value iteration, uses epsilon-greedy policy
%epsilon decides how far to stay away from the obstacle
%   P(SxSxA), R(SxSxA), discount [0,1], epsilon[0,1]
err = 0.01;

dim = size(P);
S = dim(1);
A = dim(3);

value_func_pre = zeros(S,1);
while true
    temp = zeros(S,A);
    for i=1:S
        temp = temp + squeeze(P(:,i,:))*value_func_pre(i);
    end
    q = R + discount*temp;
    value_func = epsilon*mean(q,2) + (1-epsilon)*max(q,[],2);
    if sum(abs(value_func - value_func_pre)) < err
        break;
    end
    value_func_pre = value_func;
end

[M, policy] = max(q,[],2);

end

