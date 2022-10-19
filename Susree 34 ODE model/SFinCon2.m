function y = SFinCon2

% global inpcon;

k13  = 4.381;%4.7984;            %nM^{-1}s^{-1}   %% binding of Xa with TFPI (modified 6/20/2016)
k14  = 5.293e-08;%7.427e-08;        %s^{-1}          %% dissociation of Xa:TFPI
k15  = 0.05;             %nM^{-1}s^{-1}   %% Xa:TFPI inactivation of TF:VIIa
k16  = 1.83e-04/60.0;    %%(fminc)  %nM^{-1}s^{-1}   %% ATIII inactivation of Xa % (1.83e-04/60.0,Wb,2003); (1.5e-06, HM 2002); AM,2008 0.347/60.0 (Panteleev 2006)
k17  = 1.34e-05/60.0;    %nM^{-1}s^{-1}   %% ATIII inactivation of IXa % (1.34e-05/60.0,Wb,2003); (4.9e-07, HM 2002); AM,2008 0.0162/60.0 (Panteleev 2006)
k18  = 1.79e-04;         %%(fminc)   %nM^{-1}s^{-1}   %% ATIII inactivation of IIa % (2.89e-04/60.0,Wb,2003); (7.1e-06, HM 2002); AM,2008 0.0119 11.56e-03 (Panteleev 2006)(modified 6/20/2016)
k19  = 4.5e-07;          %nM^{-1}s^{-1}   %% ATIII inactivation of TF:VIIa (HM 2002 2.3e-07)(Lawson et al.,1993,4.5e-07 no HS, 5.6e-06 with HS)
k20  = 0.01;             %nM^{-1}s^{-1}   %% binding of IXa^{m} and VIIIa^{m} (modified 6/20/2016)
k21  = 5.0e-03;          %s^{-1}          %% dissociation of IXa{m}:VIIIa{m} 0.01;%
k22  = 500.0/60.0;       %s^{-1}          %% IXa:VIIIa activation of X^{m}(2391.0/60.0 AM 2008) (Mann et al 1990) (20, KnF)
k23  = 63.0;             %nM              %% IXa:VIIIa activation of X^{m}( %160.0 AM 2008)  (Mann et al 1990) (160, KnF)
k24  = 0.4;              %nM^{-1}s^{-1}   %% binding of Xa^{m} and Va^{m} 0.1;%
k25  = 0.2;              %s^{-1}          %% dissociation of Xa:Va  0.01;%
k26  = 1344.0/60.0;      %s^{-1}          %% Xa:Va activation of II^{m}(AM 2008)  (1800.0/60.0; % (Krishnaswamy et al 1990)) Revert to original (16/9/15) (30, KnF)
k27  = 1060.0;           %nM              %% Xa:Va activation of II^{m}(AM 2008) (1000.0; % (Krishnaswamy et al 1990)) (300, KnF)
k28  = 0.023;            %s^{-1}          %% Xa^{m}-activation of VIII^{m}
K28M = 20.0;             %nM              %% Xa^{m}-activation of VIII^{m}(KnF,2001)
h8   = 0.0037;           %s^{-1}          %% spontaneous decay of VIIIa
k29  = 0.046;            %s^{-1}          %% Xa^{m}-activation of V^{m}
K29M = 10.4;             %nM              %% Xa^{m}-activation of V^{m}(KnF,2001)
h5   = 0.0028;           %s^{-1}          %% spontaneous decay of Va
k30  = 0.9;              %s^{-1}          %% IIa^{m} activation of VIII^{m}(AM,2008)(modified 6/20/2016)
K30M = 147.0;            %nM
k31  = 0.233;             %s^{-1}          %% IIa^{m} activation of V^{m}(AM,2008)(modified 6/20/2016)
K31M = 71.7;             %nM

%Dimensional Values
y = [k13; k14; k15; k16; k17; k18; k19; k20; k21; k22; k23; k24; k25;...
    k26; k27; k28; K28M; h8; k29; K29M; h5; k30; K30M; k31; K31M]; %length=25

end