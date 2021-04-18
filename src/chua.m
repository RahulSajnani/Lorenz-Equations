%chuaplot
%Creates a plot of a Chua Circuit and the Chua Diode function
%%%%%%%%%%%%%%%%%%%%
% COMPONENT VALUES
R = 2500; %Ohms
C1=10*10^-9;%F
C2=100*10^-9;%F
L=0.375;%H
%Chua's Diode paramteres
m0=-1.1;
m1=-0.7;

% Wires and Window
plot([-3;3],[-1;-1],'black')

hold
plot([-3;-1],[1;1],'black')
plot([1;3],[1;1],'black')
xlim([-5 5])
ylim([-2 2])
set(gca,'xtick',[])
set(gca,'ytick',[])
title('Chua Circuit')
%Resistor
plot([-1;-0.8],[1;1],'black')
plot([0.8;1],[1;1],'black')
t = -0.8:1/1000:0.8;
r = 0.1*sawtooth(2*pi*3*t,0.5)+0.95;
plot(t,r,'black')
text(-0.5,1.2,strcat('R = ',num2str(R),' \Omega'))
%Capaciors
%C1
plot([1;1],[1;.05],'black')
plot([1;1],[-.05;-1],'black')
plot([.7;1.3],[.05;.05],'black')
plot([.7;1.3],[-.05;-.05],'black')
text(1.1,-.2,strcat('C1=',num2str(C1),'F'))
%C2
plot([-1;-1],[1;.05],'black')
plot([-1;-1],[-.05;-1],'black')
plot([-.7;-1.3],[.05;.05],'black')
plot([-.7;-1.3],[-.05;-.05],'black')
text(-0.9,-.2,strcat('C2=',num2str(C2),'F'))
%Inductor
plot([-3;-3],[1;.8],'black')
plot([-3;-3],[-0.8;-1],'black')
xi= -0.2*abs(cos(13.7*t))-3;
yi= t;
plot(xi,yi,'black')
text(-3.8,-1.15,strcat('L=',num2str(L),'H'))
%Chua's Diode
plot([3;3],[1;.8],'black')
plot([3;3],[-0.8;-1],'black')
plot([2.7;3.3],[0.8;0.8],'black')
plot([2.7;3.3],[-0.8;-0.8],'black')
plot([2.7;2.7],[-0.8;0.8],'black')
plot([3.3;3.3],[-0.8;0.8],'black')
plot(xi,yi,'black')
plot(xi,yi,'black')
c = newline;
text(-3.8,-1.15,strcat('L=',num2str(L),'H'))
text(3.35,0,'Chua Diode')
text(3.8,-.2,'h(x)')
figure
syms x
h(x) = m0*x+0.5*(m1 - m0)*(abs(x+1)-abs(x-1));
fplot(h)
grid on;
title('Chua Diode function h(x)')
xlabel('Voltage (V)')
ylabel('Current (A)')