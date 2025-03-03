# region imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ===============================
# Function Definitions
# ===============================

def ode_system(t, X, *params):
    """
    Defines the system of ordinary differential equations (ODEs) for a piston system.

    Parameters:
        t (float): Time variable.
        X (list): State variables [x, xdot, p1, p2].
        params (tuple): System parameters (A, Cd, ps, pa, V, beta, rho, Kvalve, m, y).

    Returns:
        list: Time derivatives [xdot, xddot, p1dot, p2dot].
    """
    A, Cd, ps, pa, V, beta, rho, Kvalve, m, y = params
    x, xdot, p1, p2 = X
    xddot = (A * (p1 - p2)) / m  # Acceleration of the piston
    p1dot = beta / V * (Kvalve * (ps - p1))  # Pressure change in chamber 1
    p2dot = beta / V * (Kvalve * (p2 - pa))  # Pressure change in chamber 2
    return [xdot, xddot, p1dot, p2dot]


# ===============================
# Main Execution Function
# ===============================

def main():
    """
    Main function to set up and solve the ODE system, then plot the results.
    """
    # Define time span for simulation
    t_span = (0, 0.02)
    t_eval = np.linspace(0, 0.02, 300)  # More points for smoother curves

    # Define system parameters
    myargs = (4.909E-4, 0.6, 1.4E7, 1.0E5, 1.473E-4, 2.0E9, 850.0, 2.0E-5, 30, 0.002)
    pa = myargs[3]  # Ambient pressure
    ic = [0, 0, pa, pa]  # Initial conditions [position, velocity, pressure1, pressure2]

    # Solve the ODE system using the Runge-Kutta method
    sln = solve_ivp(ode_system, t_span, ic, args=myargs, t_eval=t_eval, method='RK45', rtol=1e-6, atol=1e-8)

    # Extract solution values
    t, xvals, xdot, p1, p2 = sln.t, sln.y[0], sln.y[1], sln.y[2], sln.y[3]

    # ===============================
    # Plot Results
    # ===============================
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot position and velocity
    ax1 = axs[0]
    ax1.set_title('Position and Velocity Over Time', fontsize=12)
    ax1.plot(t, xvals, 'r-', label='$x$', linewidth=1)
    ax1.set_ylabel('$x$', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(t, xdot, 'b-', label='$\dot{x}$', linewidth=1)
    ax2.set_ylabel('$\dot{x}$', fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)

    # Plot pressure values
    ax3 = axs[1]
    ax3.set_title('Pressure Over Time', fontsize=12)
    ax3.plot(t, p1, 'b-', label='$P_1$', linewidth=1)
    ax3.plot(t, p2, 'r-', label='$P_2$', linewidth=1)
    ax3.set_xlabel('Time, s', fontsize=12)
    ax3.set_ylabel('$P_1, P_2$ (Pa)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()


# ===============================
# Execute Script
# ===============================

if __name__ == "__main__":
    main()