function [dist]=pathLossModel(par,Pt,Pr,gamma,LOS)

ht=5; hr=1.65; dc=4*pi*ht*hr/(3/59); %dc=2039m, ht=BS, hr=MS

if strcmp(par.PL,'log-distance')
    dist=(10.^((Pt-Pr)./10).*(3/59/4/pi)^2).^(1/gamma); 
%     PL=((Pt-Pr));
%     A=10*gamma;B=10*gamma*log10((4*pi*59)/3);
%     dist=10.^((PL-B)/A);
elseif strcmp(par.PL,'urban')
    dist=(10.^((Pt-Pr-68.5)./10)).^(1/1.61);
elseif strcmp(par.PL,'FS')
    dist=(10.^((Pt-Pr)./10).*(3/59/4/pi)^2).^(1/2); 
elseif strcmp(par.PL,'TRG')
    dist=(10.^((Pt-Pr)./10).*ht.^2*hr.^2).^(1/4); 
elseif strcmp(par.PL,'COST-Hata') % hr=1->10; ht=30->200; f=1.5-2GHz 
    C=3; % constant C=3dB for metropolitan, C=0dB for suburban & urban
    CF= (1.1*log10(5900)-0.7)*hr + (1.56*log10(5900)-0.8); % correction factor
%     dist=60/1000; %in km
%     PL = 46.3 + 33.9*log10(5900) - 13.82*log10(hr) - CF + C +(44.9 - 6.55*log10(ht))*log10(dist);  Pr=Pt-PL;
    PL = 46.3 + 33.9*log10(5900) - 13.82*log10(hr) - CF + C; 
    y=44.9 - 6.55*log10(ht);
    dist= 10.^(-(Pr-Pt+PL)./y).*1000; %in meter
    % (improvement to Okumura-Hata model for higher freq) : PL=69.55 +26.16*..., C (urban, suburb, open-quasi, open)
elseif strcmp(par.PL,'COST-WI') % Walfisch-Ikegami
    if LOS==1 
%         PL=42.6+20*log10(5900)+26*log10(dist/1000);
        PL=42.6+20*log10(5900);
        y=26;
        dist=10.^(-(Pr-Pt+PL)./y).*1000;
    else % NLOS (FS+multiscreen+roofToStreet)
%         FS=32.4+20*log10(5900)+20*log10(dist/1000);
        FS=32.4+20*log10(5900); 
        
        AvgHRoof=20;
        if ht>AvgHRoof
            arg = 1+ht - AvgHRoof;
            Lbsh = -18*log10(arg);
            ka = 54;
            kd = 18;
        else % ht<AvgHRoof
            Lbsh = 0;
            kd = 18 - 15*(ht-AvgHRoof)/AvgHRoof;
%             ka = 54 -0.8*(ht-AvgHRoof)*dist/1000/0.5;
            ka=  54;
        end
        kf = -4.0 + 0.7 * (5900/925 - 1); %  urban=0.7, metropolitan=1.5                      
        buildingSep=40; % building separation
%         multiscreen= Lbsh + kf*log10(5900) - 9*log10(buildingSep) + ka + kd*log10(dist/1000);
        multiscreen= Lbsh + kf*log10(5900) - 9*log10(buildingSep) + ka;
        
        incidentAngle=45;  % angle:0-90deg
        if incidentAngle>=0 && incidentAngle<35
            lcri = -10 + 0.354 * incidentAngle;
        elseif incidentAngle>=35 && incidentAngle<55
            lcri = 2.5 + 0.075 * (incidentAngle - 35);
        else % incidentAngle=55->90
            lcri = 4 - 0.114 * (incidentAngle - 55);
        end
        streetWidth=10;
        temp=AvgHRoof-hr;
        roofToStreet= -16.9 - 10*log10(streetWidth) + 10*log10(5900) + 20*log10(temp) + lcri;
        
%         PL=FS+multiscreen+roofToStreet;
        PL=FS+multiscreen+roofToStreet;
        
        if ht>AvgHRoof
            y=20+kd;
            dist=10^(-(Pr-Pt+PL)/y)*1000;
        else
%             Pr=Pt-PL;
            temp=-(Pr-Pt+PL);
%             x=-0.8*(ht-AvgHRoof)*dist/1000/0.5 +20*log10(dist/1000) + kd*log10(dist/1000);
            y1=-0.8*(ht-AvgHRoof)/0.5;
            y2=20+kd;
            for ii=1:length(temp)
%                 dist(ii) = fzero(@(d) costWI_NLOS(d,temp(ii),y1,y2),0.1);
%                 dist(ii)=dist(ii)*1000;
            end
        end
    end
end

% if strcmp(par.PL,'log-distance') 
%     for ii=1:length(dist)
%         if dist(ii)>dc 
%             dist(ii)=(10.^((Pt-Pr(ii))/10).*ht.^2*hr.^2).^(1/4);
%         end
%     end
% end
