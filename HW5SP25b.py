# region imports
import HW5SP25a as pta  # Import external module for friction factor calculations and plotting
import random as rnd  # Import random module for probabilistic calculations
import numpy as np
from matplotlib import pyplot as plt

# region Global Variables
moody_fig, moody_ax = plt.subplots()  # Create global Moody diagram figure


# endregion

# region Function Definitions
def calculate_friction_factor(Re, rr):
    """
    Computes the friction factor based on the Reynolds number (Re) and relative roughness (rr).

    The function determines the friction factor using different methods based on the flow regime:
    - If Re >= 4000, it uses the Colebrook equation for turbulent flow.
    - If Re <= 2000, it uses the laminar flow equation f = 64 / Re.
    - For 2000 < Re < 4000, it interpolates f between the laminar and turbulent predictions,
      introducing some randomness using a normal distribution.

    Parameters:
        Re (float): The Reynolds number.
        rr (float): The relative roughness.

    Returns:
        float: The computed friction factor.
    """
    if Re >= 4000:
        return pta.ff(Re, rr, CBEQN=True)  # Turbulent flow (Colebrook equation)
    if Re <= 2000:
        return 64 / Re  # Laminar flow equation

    # Transition region: Compute friction factors for both regimes
    f_CB = pta.ff(Re, rr, CBEQN=True)  # Colebrook prediction
    f_lam = 64 / Re  # Laminar prediction

    # Interpolate friction factor
    mu_f = f_lam + (f_CB - f_lam) * ((Re - 2000) / 2000)
    sigma_f = 0.2 * mu_f
    return rnd.normalvariate(mu_f, sigma_f)  # Generate random friction factor from normal distribution


def compute_head_loss(f, Q, D):
    """
    Computes the head loss per foot (hf/L) using the Darcy-Weisbach equation.

    Parameters:
        f (float): Friction factor.
        Q (float): Flow rate in gallons per minute.
        D (float): Pipe diameter in inches.

    Returns:
        float: The head loss per foot in appropriate English units.
    """
    g = 32.174  # Gravity in ft/s^2
    D_ft = D / 12  # Convert diameter to feet
    V = (Q / 448.831) / (np.pi * (D_ft / 2) ** 2)  # Velocity in ft/s
    return (f * (V ** 2) / (2 * g * D_ft))  # Compute hf/L


def plot_moody_diagram(Re, f):
    """
    Updates the global Moody diagram with a new friction factor point.

    Parameters:
        Re (float): Reynolds number.
        f (float): Computed friction factor.

    Returns:
        None
    """
    global moody_ax
    marker = '^' if 2000 < Re < 4000 else 'o'  # Triangle for transition, circle otherwise
    moody_ax.scatter(Re, f, marker=marker, color='red')
    moody_ax.set_xlabel("Reynolds Number (Re)")
    moody_ax.set_ylabel("Friction Factor (f)")
    moody_ax.set_title("Moody Diagram")
    plt.draw()


def main():
    """
    Main function to compute and plot the friction factor based on user input.

    Steps:
    - Prompts the user for pipe diameter, roughness, and flow rate.
    - Computes Reynolds number and relative roughness.
    - Computes the friction factor and head loss per foot.
    - Updates the Moody diagram with each new input set.
    - Allows the user to input multiple cases without resetting the diagram.

    Returns:
        None
    """
    while True:
        D = float(input("Enter pipe diameter (in inches): ") or 12)
        e_mics = float(input("Enter pipe roughness (in micro-inches): ") or 150)
        Q = float(input("Enter flow rate (in gallons per minute): ") or 500)

        # Convert roughness from micro-inches to feet
        e = e_mics * 1e-6 / 12
        rr = e / (D / 12)  # Compute relative roughness

        # Compute Reynolds number
        v_kinematic = 1.08e-5  # Kinematic viscosity of water in ft^2/s
        V = (Q / 448.831) / (np.pi * (D / 12 / 2) ** 2)  # Velocity in ft/s
        Re = (V * (D / 12)) / v_kinematic  # Reynolds number

        # Compute friction factor and head loss
        f = calculate_friction_factor(Re, rr)
        hf_L = compute_head_loss(f, Q, D)

        # Display results
        print(f"Reynolds Number: {Re:.2f}")
        print(f"Friction Factor: {f:.5f}")
        print(f"Head Loss per Foot (hf/L): {hf_L:.5f} ft/ft")

        # Plot the new point on the Moody diagram
        plot_moody_diagram(Re, f)
        plt.pause(0.1)  # Allow the plot to update

        # Ask the user if they want to input another set
        cont = input("Would you like to enter another set? (y/n): ").strip().lower()
        if cont != 'y':
            break
    plt.show()


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion
