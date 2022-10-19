function y = SFinCon3

b11 = 0.01;     % nM^{-1}s^{-1}   %% platelet-binding of fXI/XIa; FOGELSON et al., 2011
d11 = 0.1;      % sec^{-1}        %% dissociation of fXI/XIa from platelets
b10 = 0.029;    % nM^{-1}s^{-1}   %% platelet-binding of fX/Xa (Krishnaswamy et al. 1998)
d10 = 3.3;      % sec^{-1}        %% dissociation of fX/Xa from platelets
b2  = 0.01;     % nM^{-1}s^{-1}   %% platelet-binding of fII/IIa
d2  = 5.9;      % sec^{-1}        %% dissociation of fII/IIa from platelets
b9  = 0.01;     % nM^{-1}s^{-1}   %% platelet-binding of fIX/IXa
d9  = 0.0257;   % sec^{-1}        %% dissociation of fIX/IXa from platelets
b8  = 4.3e-03;  % nM^{-1}s^{-1}   %% platelet-binding of fVIII/VIIIa (Raut et al 1999)
d8  = 2.46e-03; % sec^{-1}        %% dissociation of fVIII/VIIIa from platelets (Raut et al 1999)
b5  = 0.057;    % nM^{-1}s^{-1}   %% platelet-binding of fV/Va (Krishnaswamy et al. 1998)
d5  = 0.17;     % sec^{-1}        %% dissociation of fV/Va from platelets
kpp = 0.3;      % nM^{-1}s^{-1}   %% platelet-activation of platelet (KnF 2001)
kp2 = 0.37;     % s^{-1}          %% thrombin-activation of platelet 5.4/60.0;%

%Dimensional values
y = [b11; d11; b10; d10; b2; d2; b9; d9; b8; d8; b5; d5; kpp; kp2]; %Length = 14

end