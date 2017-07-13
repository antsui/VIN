addpath(genpath('./MDPtoolbox/MDPtoolbox'));
load('data/gridworld_28.mat');

discount = 0.9;
epsilon = 0.1;
imsize = 28;

% generate reward function
obstacle = 1-im_data(1,:);
goal = value_data(1,:);
reward = zeros(size(goal));
reward(obstacle == 1) = -100;
reward(goal == 10) = 10;
reward = reshape(reward,[imsize,imsize])';

state_space = (0:imsize^2-1)';
action_space = 0:8-1;
P = zeros(length(state_space), length(state_space), length(action_space));
R = zeros(length(state_space), length(action_space));
action_vects = [-1,0; 1,0; 0,1; 0,-1; -1,1; -1,-1; 1,1; 1,-1];

state_pos = [floor(state_space/imsize), mod(state_space,imsize)];
for s = 0:length(state_space)-1
    for a = 0:length(action_space)-1
        next_state_pos = [state_pos(s+1,1)+action_vects(a+1,1),...
            state_pos(s+1,2)+action_vects(a+1,2)];
        if ~(next_state_pos(1)>=0 && next_state_pos(1)<=imsize-1 && ...
                next_state_pos(2)>=0 && next_state_pos(2)<=imsize-1)
            next_state = s;
        else
            next_state = next_state_pos(1)*imsize + next_state_pos(2);
        end
        next_state_pos = [floor(next_state/imsize), mod(next_state,imsize)];
        P(s+1,next_state+1,a+1) = 1;
        R(s+1,a+1) = reward(next_state_pos(1)+1, next_state_pos(2)+1) - sqrt(action_vects(a+1,1)^2+action_vects(a+1,2)^2); % every step leads to a cost
    end
end

% [policy, iter, cpu_time] = mdp_value_iteration (P, R, discount);
[ policy2 ] = epsilon_greedy_value_iteration( P, R, discount, 0 );
[ policy2 ] = epsilon_greedy_value_iteration( P, R, discount, 0.025 );
display('policy generated');

%% generate path
start = randi(length(state_space));
start = 733;
goal_state = find(goal==10);

path1 = [start];
while path1(end)~=goal_state
    a = policy1(path1(end));
    next_state = find(P(path1(end), :, a) == 1);
    path1 = [path1;next_state];
end
path1_pos = [floor(path1/imsize)+1,mod(path1-1,imsize)+1];

path2 = [start];
while path2(end)~=goal_state
    a = policy2(path2(end));
    next_state = find(P(path2(end), :, a) == 1);
    path2 = [path2;next_state];
end
path2_pos = [floor(path2/imsize)+1,mod(path2-1,imsize)+1];

%% plot
figure();
img = reshape(1-obstacle,[imsize,imsize])*255;
imshow(img,'InitialMagnification','fit');
hold on;
plot(path1_pos(:,1),path1_pos(:,2));
plot(path1_pos(end,1),path1_pos(end,2),'*');
plot(path1_pos(1,1),path1_pos(1,2),'o');
legend('path','goal','start');
plot(path2_pos(:,1),path2_pos(:,2));