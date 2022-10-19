function [c,ceq] = simple_constraint(inpcon)

format long

global IC C1 C2 C3 NDM TIM;

%Y = [TF; VII; TF:VII; VIIa; TF:VIIa; IX; IXa; IX(m); IXa(m); X; Xa;...
%     X(m); Xa(m); II; IIa; II(m); IIa(m); PL; AP; VIII; VIIIa;...
%     VIII(m); VIIIa(m); IXa:VIIIa(m); V; Va; V(m); Va(m); Xa:Va(m); I; Ia;
%     TFPI; Xa:TFPI;...
%     ATIII]; % U=34

IC = [0.005; 10.0; 0.0; 0.001; 0.0; 90.0; 0.009; 0.0; 0.0; 170.0; 0.017;...
      0.0; 0.0; 1400.0; 0.14; 0.0; 0.0; 10.0; 0.0010; 0.7; 0.00007;...
      0.0; 0.0; 0.0; 20.0; 0.002; 0.0; 0.0; 0.0; 7000.0; 0.70; 2.5; 0.0;...
      3400.0]; %nM 

% 0.01% activation;
  
NDM = [0.005; 10.0; 0.005; 10.0; 0.005; 90.0; 90.0; 10.0; 10.0; 170.0;...
       170.0; 10.0; 10.0; 1400.0; 1400.0; 10.0; 10.0; 10.0; 10.0;...
       0.7; 0.7; 0.7; 0.7; 0.7; 20.0; 20.0; 10.0; 10.0; 10.0; 7000.0;...
       7000.0; 2.5; 2.5; 3400.0]; %nM

C1  = SFinCon1;
C2  = SFinCon2;
C3  = SFinCon3;
% C4  = FinCon4;
TIM = 1800;%690.071; 
inpcon
C2(1) = inpcon(1)
C2(6) = inpcon(2)
C2(8) = inpcon(3)
errmin = 0.0;

% C41=0.009;
% C42=5e-07;
% k = [k(1); k(2)];

ceq=[];

exp_t = [129.132 149.149 169.156 189.20 212.347 230.948 251.171 271.412...
        290.023 309.946 331.255 349.473 372.366 387.613 407.517 430.494...
        448.899 468.831 491.884 510.298 528.721 551.774 570.225 591.722... 
        613.238 628.607 650.113 673.166 690.071];

tspan = exp_t./TIM; %[0:1:1800]./TIM;                            %non-dimensionalized time;
xinit = IC./NDM;                                    %non-dimensionalized input conc.;

exp_IIa = ([1.78e-08; 3.71e-08; 5.15e-08; 8.53e-08; 1.34e-07; 2.17e-07;...
          3.43e-07; 4.79e-07; 5.67e-07; 5.37e-07; 4.30e-07; 3.13e-07;...
          2.29e-07; 1.66e-07; 1.26e-07; 8.71e-08; 6.73e-08; 4.27e-08;...
          4.24e-08; 2.75e-08; 1.75e-08; 1.72e-08; 2.19e-08; 1.18e-08;...
          1.15e-08; 1.13e-08; 6.18e-09; 5.88e-09; 5.66e-09])*10^(+9);

    [t,Y] = ode15s(@SFinclot,tspan,xinit);
    spec = Y<0;                                         %to eliminate negative zeroes in output;
    Y(spec)= 0;
% Final(C42);
    a = (Y(:,15)*NDM(15))
    b = (Y(:,17)*NDM(17))
    totIIa = (a+b)
    errvec = ((totIIa - exp_IIa)./exp_IIa).^2
    errmin = sum(errvec)
   
    end
    