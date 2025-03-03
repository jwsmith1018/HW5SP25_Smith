# region imports
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# endregion

# region functions
def ff(Re, rr, CBEQN=False):
    """
    Calculates the Darcy-Weisbach friction factor for pipe flow.

    Parameters:
    Re (float): Reynolds number.
    rr (float): Relative pipe roughness (between 0 and 0.05).
    CBEQN (bool): If True, uses the Colebrook equation; otherwise, uses the laminar equation.

    Returns:
    float: Friction factor.
    """
    if CBEQN:
        # Colebrook equation as an implicit function for fsolve
        cb = lambda f: 1 / np.sqrt(f) + 2.0 * np.log10(rr / 3.7 + 2.51 / (Re * np.sqrt(f)))
        result = fsolve(cb, 0.02)  # Initial guess for fsolve is 0.02
        return result[0]
    else:
        return 64 / Re  # Laminar flow equation


def plotMoody(plotPoint=False, pt=(0, 0)):
    """
    Generates a Moody diagram plotting the friction factor as a function of Reynolds number.

    Parameters:
    plotPoint (bool): If True, plots a specific point on the graph.
    pt (tuple): Coordinates (Re, f) of the point to be plotted.

    Returns:
    None
    """
    # Step 1: Create log-spaced arrays for different Reynolds number ranges
    ReValsCB = np.logspace(np.log10(4000), np.log10(1e8), 100)  # Turbulent range
    ReValsL = np.logspace(np.log10(600), np.log10(2000), 20)  # Laminar range
    ReValsTrans = np.logspace(np.log10(2000), np.log10(4000), 20)  # Transition range

    # Step 2: Create array for range of relative roughness values
    rrVals = np.array([0, 1E-6, 5E-6, 1E-5, 5E-5, 1E-4, 2E-4, 4E-4, 6E-4, 8E-4,
                       1E-3, 2E-3, 4E-3, 6E-3, 8E-3, 1.5E-2, 2E-2, 3E-2, 4E-2, 5E-2])

    # Step 3: Calculate friction factor values
    ffLam = np.array([ff(Re, 0, False) for Re in ReValsL])  # Laminar range
    ffTrans = np.array([ff(Re, 0, False) for Re in ReValsTrans])  # Transition range
    ffCB = np.array([[ff(Re, relRough, True) for Re in ReValsCB] for relRough in rrVals])  # Turbulent range

    # Step 4: Construct the plot
    plt.figure(figsize=(10, 6))
    plt.loglog(ReValsL, ffLam, 'b-', label='Laminar Flow')  # Solid line for laminar flow
    plt.loglog(ReValsTrans, ffTrans, 'b--', label='Transition Flow')  # Dashed line for transition

    for nRelR in range(len(ffCB)):
        plt.loglog(ReValsCB, ffCB[nRelR], 'k')  # Turbulent flow for different roughnesses
        plt.annotate(f'{rrVals[nRelR]:.0e}', xy=(ReValsCB[-1], ffCB[nRelR, -1]))

    # Formatting
    plt.xlim(600, 1e8)
    plt.ylim(0.008, 0.10)
    plt.xlabel(r"Reynolds number $Re$")
    plt.ylabel(r"Friction factor $f$")
    plt.text(2.5E8, 0.02, r"Relative roughness $rac{\epsilon}{d}$", rotation=90)

    ax = plt.gca()
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=12)
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3f"))
    plt.grid(which='both')

    if plotPoint:
        plt.plot(pt[0], pt[1], 'ro', markersize=8, markeredgecolor='red', markerfacecolor='none')

    plt.legend()
    plt.show()


def main():
    """
    Main function to generate the Moody diagram.
    """
    plotMoody(plotPoint=False)  # Ensure the function is called properly


# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion