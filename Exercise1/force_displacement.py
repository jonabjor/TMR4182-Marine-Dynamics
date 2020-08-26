import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Plot parameters
plt.title("Resulting plot for static experiment")
plt.xlabel("Displacement [m]")
plt.ylabel("Force [N]")

# Plotting the resulting points
F = np.array([0, 1.962, 3.924, 5.886, 7.848, 9.81])  # Force [N]
x = np.array([0, 0.011, 0.022, 0.033, 0.045, 0.056])  # Displacement [m]

plt.plot(x, F, marker="x", linestyle="none")

# Linear regression, and plotting the best solution
slope, intercept, r, p, std_err = stats.linregress(x, F)
plt.plot(x, intercept + x*slope, color="orange")

# Show plot at last
plt.show()
