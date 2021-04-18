import numpy as np
import matplotlib.pyplot as plt

class chaos_equations:
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

    def plotBifurcation(self):
        '''
        Plot bifurcations
        '''
        rho_init = self.rho
        critical_points_locations = {"p_0":[], "p_1":[], "p_2": []}
        critical_points_eigenvalues = {"p_0":[], "p_1":[], "p_2": []}
        rho_locations = []
        
        rho_high = 26.5
        for rho in np.linspace(0, rho_high, 1000):
            self.rho = rho
            rho_locations.append(rho)
            critical_points_dict = self.getCriticalPoints()
            for key in critical_points_locations:
                if critical_points_dict.get(key) is None:
                    critical_points_locations[key].append(np.array([0, 0, 0]))
                    critical_points_eigenvalues[key].append(np.array([0, 0, 0]))
                else:
                    critical_points_locations[key].append(critical_points_dict[key])
                    eigenvalues, eigenvectors = self.getStabilityPoint(critical_points_dict[key])
                    # print(np.real(eigenvalues), np.imag(eigenvalues))
                    critical_points_eigenvalues[key].append(eigenvalues)

        rho_locations = np.array(rho_locations)

        minor_ticks = np.arange(0, rho_high // 1, 1)
        
        # Plot Pitchfork bifurcation
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        i = 0
        for axis in ["x", "y", "z"]:
            for key in critical_points_locations:
                
                trajectory = np.stack(critical_points_locations[key], axis = 0)
                # print(trajectory.shape)
                axs[i].plot(rho_locations[:trajectory.shape[0]], trajectory[:, i], label = key)
                axs[i].set_title("%s vs. rho" % axis)
                axs[i].set_xlabel("rho")
                axs[i].set_ylabel(axis)   
                axs[i].set_xticks(minor_ticks, minor = True) 
                axs[i].grid(which='minor', alpha=0.2)
                axs[i].grid(True)
                axs[i].legend()
            i = i + 1
        
        plt.suptitle("Pitchfork bifurcation by varying value of rho")
        plt.show()


        fig = plt.figure(constrained_layout = True)
        m = 1
        # Plot Hopf bifurcation
        for key in critical_points_locations:
            ax = fig.add_subplot(1, 3, m, projection='3d')  
            m = m + 1
            i = 0
            for axis in ["eigenvalue 1", "eigenvalue 2", "eigenvalue 3"]:    
                trajectory = np.stack(critical_points_eigenvalues[key], axis = 0)
                ax.plot3D(rho_locations, np.real(trajectory[:, i]), np.imag(trajectory[:, i]), label = axis)
                ax.set_xlabel("rho")
                ax.set_ylabel("a")   
                ax.set_zlabel("b")   
                title = "Eigenvalue vs. rho for %s" % key
                ax.set_title(title)
                ax.grid(True)
                i = i + 1
            ax.legend()
        
        plt.suptitle("Hopf bifurcation by varying value of rho. a + ib vs. rho => (a, b, rho)")
        # plt.legend()
        plt.show()

        self.rho = rho_init

    def getLorenzMatrix(self, x, y, z):
        '''
        Get Lorenz matrix dX/dt = AX
        
        dx/dt = -sigma * (x+g) + sigma * y + 0 * z
        dy/dt = x + (- 1) * y +z = x -y + z
        dz/dt = 0 * x - rho * y + 0 * z = - rho * y
        '''
        A = np.array([[        -self.sigma,     self.sigma,            0], 
                      [                  1,             -1,            1],
                      [              0,          -self.rho,            0]])
        
        return A

    def getLorenzTrajectory(self, initial_point, n,m,num_points = 5000):
        '''
        Get lorenz trajectory given initial point
        '''

        X = np.array(initial_point)
        A = self.getLorenzMatrix(X[0], X[1], X[2])
        
        trajectory = [X]
        for i in range(num_points):
            
            g = n*X[0]+0.5*(m-n)*(np.abs(X[0]+1)-np.abs(X[0]-1))
            delta_X = A @ X * self.delta_t
            X = delta_X + X
            X[0] = X[0] -self.sigma * g * self.delta_t
            A = self.getLorenzMatrix(X[0], X[1], X[2])
            trajectory.append(X)
        
        trajectory = np.stack(trajectory, axis = 0)

        return trajectory

    def plotLorenzTrajectory(self, initial_point, n,m,num_points = 5000):
        '''
        Plot the lorenz trajectory given initial point
        '''
        critical_points = self.getCriticalPoints()
        trajectory = self.getLorenzTrajectory(initial_point, n,m,num_points)

        ax = plt.figure().add_subplot(projection='3d')          
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Chua circuit equations live plot. Init point = (%.02f, %.02f, %.02f)" % (initial_point[0], initial_point[1], initial_point[2]))
        
        ax.plot3D(initial_point[0], initial_point[1], initial_point[2], "ro")
        # for p in critical_points:
        #     point = critical_points[p]
        
        #     if not np.iscomplex(point).any():
        #         # eigenvalues, eigenvectors = self.getStabilityPoint(point)

        #         ax.plot3D(point[0], point[1], point[2], "go")

        plot_steps = 100
        for i in range(0, trajectory.shape[0], plot_steps):
            ax.plot3D(trajectory[i: i + plot_steps + 1, 0], trajectory[i: i + plot_steps + 1, 1], trajectory[i:i + plot_steps + 1, 2], "b")
            plt.pause(0.2 /  (num_points / plot_steps))
            
        plt.show()
    
    def plotLorenzAlongAxis(self, initial_point, n,m,num_points = 5000):
        '''
        plot x, y, z w.r.t t
        '''
        trajectory = self.getLorenzTrajectory(initial_point,n,m ,num_points)
        t = np.linspace(0, trajectory.shape[0] * self.delta_t, trajectory.shape[0]) 
        minor_ticks = np.arange(0, t[-1]//1, 1)
        fig, axs = plt.subplots(3, 1, constrained_layout=True)

        axs[0].plot(t, trajectory[:, 0], "b")
        axs[0].set_title("x (voltage across C1) vs. t")
        axs[0].set_xlabel("t")
        axs[0].set_ylabel("x")
        axs[0].set_xticks(minor_ticks, minor = True) 
        axs[0].grid(which='minor', alpha=0.2)
        axs[0].grid(True)
        
        axs[1].plot(t, trajectory[:, 1], "b")
        axs[1].set_title("y (voltage across C2) vs. t")
        axs[1].set_xlabel("t")
        axs[1].set_ylabel("y")
        axs[1].set_xticks(minor_ticks, minor = True) 
        axs[1].grid(which='minor', alpha=0.2)
        axs[1].grid(True)
        

        axs[2].plot(t, trajectory[:, 2], "b")
        axs[2].set_title("z (current of diode) vs. t")
        axs[2].set_xlabel("t")
        axs[2].set_ylabel("z")
        axs[2].set_xticks(minor_ticks, minor = True) 
        axs[2].grid(which='minor', alpha=0.2)
        axs[2].grid(True)
        
        
        plt.suptitle("Plot x, y, and z vs. t")
        plt.show()


if __name__ == "__main__":

    lorenz = chaos_equations(prandtl_number = 16, rayleigh_number = 30, beta = 8/3, delta_t = 1e-2)
    lorenz.plotLorenzTrajectory([1, 0, 0],-0.7, -1.2)
    # lorenz.plotLorenzAlongAxis([1, 0, 0],-0.7, -1.2)
    # lorenz.plotBifurcation()
    # critical_points = lorenz.getCriticalPoints()
    # print(critical_points)
        
