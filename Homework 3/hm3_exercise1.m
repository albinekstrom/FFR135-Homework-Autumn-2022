%% Homework 2: Classification challenge
% Albin Ekström
% Date 11 okt 2022

clc
clear variables

x = 25; y = 25;
f = 3; % kernal size
p = [1 1 1 1];
s = 2;
prevkernals = 3;
kernals = 14;

% CONV
x_c = (x+p(1)+p(2)-f)/s + 1
y_c = (y+p(3)+p(4)-f)/s + 1
trained_para_c = ((f*f*prevkernals)+1)*14

f = 2; % kernal size
p = [1 1 0 0];
s = 1;

% MAX POOLING
x_p = (x_c+p(1)+p(2)-f)/s + 1
y_p = (y_c+p(3)+p(4)-f)/s + 1

para_FC = 15;

% FC
trained_para_FC = (para_FC*(x_p*y_p*kernals)+para_FC)

para_out = 10;

% Output
trained_para_out = para_out*para_FC+para_out