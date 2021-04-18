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

    def getStabilityPoint(self, point):
        '''
        Get the stability of critical point
        '''

        Jacobian = np.array([[          -self.sigma,   self.sigma,             0], 
                             [  self.rho - point[2],           -1,     -point[0]],
                             [             point[1],     point[0],    -self.beta]])
        
        eigenvalues, eigenvectors = np.linalg.eig(Jacobian)
        return eigenvalues, eigenvectors

    def plotBifurcation(self, initial_point):
        '''
        Plot bifurcations
        - 
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

    def plotLorenzTrajectory(self, initial_point, num_points = 5000):
        '''
        Plot the lorenz trajectory given initial point
        '''
        critical_points = self.getCriticalPoints()
        trajectory = self.getLorenzTrajectory(initial_point, num_points)

        ax = plt.figure().add_subplot(projection='3d')          
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Lorenz equations live plot. Init point = (%.02f, %.02f, %.02f)" % (initial_point[0], initial_point[1], initial_point[2]))
        
        ax.plot3D(initial_point[0], initial_point[1], initial_point[2], "ro")
        for p in critical_points:
            point = critical_points[p]
        
            if not np.iscomplex(point).any():
                eigenvalues, eigenvectors = self.getStabilityPoint(point)

                ax.plot3D(point[0], point[1], point[2], "go")

        
        
        plot_steps = 100
        for i in range(0, trajectory.shape[0], plot_steps):
            ax.plot3D(trajectory[i: i + plot_steps + 1, 0], trajectory[i: i + plot_steps + 1, 1], trajectory[i:i + plot_steps + 1, 2], "b")
            plt.pause(0.2 /  (num_points / plot_steps))
            
        plt.show()
    
    def plotLorenzAlongAxis(self, initial_point, num_points = 5000):
        '''
        plot x, y, z w.r.t t
        '''
        trajectory = self.getLorenzTrajectory(initial_point, num_points)
        t = np.linspace(0, trajectory.shape[0] * self.delta_t, trajectory.shape[0]) 
        
        fig, axs = plt.subplots(3, 1, constrained_layout=True)

        axs[0].plot(t, trajectory[:, 0], "b")
        axs[0].set_title("x (convection) vs. t")
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("x")
        
        axs[1].plot(t, trajectory[:, 1], "b")
        axs[1].set_title("y (temperature difference (horizontal)) vs. t")
        axs[1].set_xlabel("t")
        axs[1].set_ylabel("y")
        

        axs[2].plot(t, trajectory[:, 2], "b")
        axs[2].set_title("z (temperature difference (vertical)) vs. t")
        axs[2].set_xlabel("t")
        axs[2].set_ylabel("z")
        
        plt.suptitle("Plot x, y, and z vs. t")
        plt.show()


if __name__ == "__main__":

    lorenz = Lorenz_equations(prandtl_number = 10, rayleigh_number = 16, beta = 8/3, delta_t = 1e-2)
    lorenz.plotLorenzTrajectory([0, -5, -1])
    lorenz.plotLorenzAlongAxis([0, -5, -1])
    critical_points = lorenz.getCriticalPoints()
    # print(critical_points)
        
