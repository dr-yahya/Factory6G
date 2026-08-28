function y=add_CP(x)
% Add CP (Cyclic Prefix) of length Ng
% xx=reshape(x,5,128);
 Ng = length(x)/4;
% w = cat(2,xx(1,:),xx(2,:),xx(3,:),xx(4,:),xx(5,:));
%x=x';
%y = [x(:,end-Ng+1:end) x zeros(1,224)];%1x1024
y = [x(:,end-Ng+1:end) x]; %1x800
%y = [w(:,end-Ng+1:end) w];