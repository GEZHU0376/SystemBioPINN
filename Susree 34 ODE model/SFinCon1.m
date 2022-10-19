function y = SFinCon1

k1   = 3.2e-03;     %nM^{-1}s^{-1}    %% binding of TF & VII 0.0458;%
k2   = 3.1e-03;     %s^{-1}           %% dissociation of TF:VII 0.0183;%
k3   = 0.023;       %nM^{-1}s^{-1}    %% binding of TF & VIIa 0.0458;%
k4   = 3.1e-03;     %s^{-1}           %% dissociation of TF:VIIa 0.0183;%
k5   = 4.4e-04;     %nM^{-1}s^{-1}    %% auto-activation of VII (H&M,2002)
k6   = 0.013;       %nM^{-1}s^{-1}    %% Xa-activation of VII
k7   = 2.3e-05;     %nM^{-1}s^{-1}    %% IIa-activation of VII
k8   = 69.0/60.0;   %s^{-1}           %% TF:VIIa activation of X (103.0/60.0; AM,2008) (Mann et al 1990,Blood J)
K8M  = 450.0;       %nM               %% TF:VIIa activation of X (240.0; AM,2008) (Mann et al 1990)
k9   = 15.6/60.0;   %s^{-1}           %% TF:VIIa activation of IX (32.4/60.0; AM,2008) (Mann et al 1990,Blood J) 
K9M  = 243.0;       %nM               %% TF:VIIa activation of IX (24.0; AM,2008) (Mann et al 1990) 
k10  = 7.5e-06;     %nM^{-1}s^{-1}    %% Xa-activation of II
k11  = 54.0/60.0;   %s^{-1}           %% IIa-activation of VIII(194.4/60.0; AM,2008) (Hill-Eubanks & Lollar 1990)(modified 6/20/2016)
K11M = 147.0;       %nM               %112000;
k12  = 0.233;   %s^{-1}           %% IIa-activation of V (%27.0/60.0; AM,2008) (Monkovic & Tracy, 1990)(modified 6/20/2016)
K12M = 71.7;        %nM               %140.5;
kf   = 59.0;        % s^{-1}          %% IIa-activation of fibrinogen (AM,2008)
KfM  = 3160.0;      % nM

%K7M  = 3200;        %nM               %% auto-activation of VII

%Dimensional Values
y = [k1; k2; k3; k4; k5; k6; k7; k8; K8M; k9; K9M; k10; k11; K11M; k12;...
    K12M; kf; KfM]; %length=18

end