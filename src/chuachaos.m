[t,y] = ode45(@chua_function,[0 150],[1 0 0]);      
%plot3(y(:,1),y(:,2),y(:,3)) %3D Plots
%title("Chua's circuit 3D")
%xlabel('x')
%ylabel('y')
%zlabel('z')
plot(t,y(:,3)) %2D Plots
title("Chua's circuit 2D")
xlabel('t')
ylabel('z')

            
function out = chua_function(t,p)
a = 16; 
B = 30;
m = -1.2; 
n = -0.7;
x = p(1);
y = p(2);
z = p(3);
g = n*x+0.5*(m-n)*(abs(x+1)-abs(x-1)); 
xdot = a*(y-x-g);
ydot = x - y+ z;
zdot = -B*y;
out = [xdot ydot zdot]';
end
