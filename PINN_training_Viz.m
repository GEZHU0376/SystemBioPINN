% PINN training time plot
T = readtable('/loss_function/depth5.csv');
T = table2array(T);
x = T(:,1); % X : training steps
y1 = T(:,2); % y1: training loss
%y2 = T(:,3); % y2: test loss


T = readtable('depth5.csv');
T = table2array(T);
T = readtable('/loss_function/depth5.csv');
T_5 = table2array(T);
T = readtable('/loss_function/depth8.csv');
T_8 = table2array(T);
T = readtable('/loss_function/depth10.csv');
T_10 = table2array(T);

t5 = T_5(1:60,2);
t5 = T_5(1:60,1:2);
t8 = T_8(1:60,1:2);
t10 = T_10(1:60,1:2);

t5_x = t5(1:35,1);
t5_y = t5(1:35,2);
t8_x = t8(1:35,1);
t8_y = t8(1:35,2);
t10_x = t10(1:35,1);
t10_y = t10(1:35,2);

hold on;
plot(t5_x,t5_y,'b');
legend('Depth = 5');

plot(t8_x,t8_y,'r');
legend('Depth = 8');
plot(t10_x,t10_y,'g');
legend('Depth = 10');
hold off;