import numpy as np
import matplotlib.pyplot as plt

class Lorenz_equations:
    '''
    Lorenz equations class
    '''

    def __init__(self, prandtl_number, rayleigh_number, beta, delta_t):
        
        self.sigma = prandtl_number
        self.rho = rayleigh_number
        self.beta = beta
        self.delta_t = delta_t
        

    def getCriticalPoints(self):
        '''
        Get critical points for Lorenz equation
        dX/dt = 0
        '''
        
        critical_points = {}
        p_0 = np.array([0, 0, 0])

        critical_points["p_0"] = p_0
        if not (self.rho - 1 < 0):
            p_1 = np.array([np.sqrt(self.beta * (self.rho - 1)),  np.sqrt(self.beta * (self.rho - 1)), self.rho - 1])
            critical_points["p_1"] = p_1
            
            p_2 = np.array([-np.sqrt(self.beta * (self.rho - 1)),  -np.sqrt(self.beta * (self.rho - 1)), self.rho - 1])
            critical_points["p_2"] = p_2
        return critical_points

    def getStabilityPoints(self, point):
        '''
        Get the stability of critical point
        '''
        pass
    
    def getLorenzMatrix(self, x, y, z):
        '''
        Get Lorenz matrix dX/dt = AX
        
        dx/dt = -sigma * x + sigma * y + 0 * z
        dy/dt = (rho - z/2) * x + (- 1) * y - (x/2) * z = rho * x -y - xz
        dz/dt = (y/2) * x + (x/2) * y - (beta) * z = xy - beta * z
        '''

        A = np.array([[        -self.sigma,     self.sigma,            0], 
                      [   self.rho - z / 2,             -1,       -x / 2],
                      [              y / 2,          x / 2,   -self.beta]])
        
        return A

    def getLorenzTrajectory(self, initial_point, num_points = 5000):
        '''
        Get lorenz trajectory given initial point
        '''

        X = np.array(initial_point)
        A = self.getLorenzMatrix(X[0], X[1], X[2])
        
        trajectory = [X]
        for i in range(num_points):
            
            delta_X = A @ X * self.delta_t
            X = delta_X + X
            A = self.getLorenzMatrix(X[0], X[1], X[2])
            trajectory.append(X)
        
        trajectory = np.stack(trajectory, axis = 0)

        return trajectory
    
    def plotLorenzTrajectory(self, initial_point, num_points = 10000):
        '''
        Plot the lorenz trajectory given initial point
        '''
        critical_points = self.getCriticalPoints()
        trajectory = self.getLorenzTrajectory(initial_point, num_points)

        ax = plt.figure().add_subplot(projection='3d')          
        for p in critical_points:
            point = critical_points[p]

            if not np.iscomplex(point).any():
                ax.scatter3D(point[0], point[1], point[2])

        ax.plot3D(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
        plt.show()


if __name__ == "__main__":

    lorenz = Lorenz_equations(prandtl_number = 10, rayleigh_number = 0, beta = 10/3, delta_t = 1e-2)
    lorenz.plotLorenzTrajectory([1, 2, 0])
    critical_points = lorenz.getCriticalPoints()
    print(critical_points)
        
