# Analysis of Convergence Issues for N=16

I've examined the `run_sphere_10k` and `run_double_star` logs and verified that you did indeed observe a higher `RMSD error` for $N=16$ than $N=8$ at 10,000 epochs.

I have isolated the exact mathematical and implementation reasons for this discrepancy. The short answer is: **the discrete finite volume scheme you implemented for Robin B.C. has a local truncation error of $O(1)$ at the cut cells, which limits the global accuracy to First-Order ($O(\Delta x)$). Because of this, the MLP learns an exact solution to a slightly inaccurate discrete system.**

### 1. The Truncation Error in Robin B.C.

In `discretization.py`, you enforce the Robin boundary condition using:
```python
alpha_ell = self.alphaRobin_integrate_over_interface_at_point(point, dx, dy, dz)
lhs += alpha_ell * u_m_ijk
```
Here, `alpha_ell` correctly calculates $\int_{\Gamma} \alpha(x) dS$ with second-order geometric accuracy. However, by multiplying it by $u_{ijk}$ (the value at the cell center), you are implicitly approximating:
$$ \int_{\Gamma \cap V_i} \alpha(x) u(x) dS \approx u_{ijk} \int_{\Gamma \cap V_i} \alpha(x) dS $$

Because the distance from the cell center $x_i$ to the interface $\Gamma$ is $O(\Delta x)$, the error in this approximation is $O(\Delta x^2)$ per cut cell. Divided by the cell volume $O(\Delta x^3)$, this introduces an $O(1)$ **local truncation error** into the PDE at the boundaries. A local truncation error of $O(1)$ on a co-dimension 1 boundary guarantees that the global scheme is strictly **First-Order Accurate** ($O(\Delta x)$). 

### 2. Why does the error INCREASE for N=16?

If the scheme is first-order, the error should go down slowly ($1/N$), but it shouldn't go *up*. Why did it go up for $N=16$ in your logs?

Look at the final loss values from your `run_sphere_10k` logs:
* **N=8 (10k epochs):** Final Discrete Loss = `9.12e-06`, RMSD = `0.00501`
* **N=16 (10k epochs):** Final Discrete Loss = `3.88e-06`, RMSD = `0.00516`

At $N=16$, the neural network actually achieved a **lower discrete residual loss**. It solved your discrete equations *better* than the $N=8$ run. However, because those discrete equations have an inherent $O(dx)$ truncation error at the boundary, the network perfectly fit a mathematically flawed system!

Furthermore, your recent commit (*"Fix u_m stencil evaluation in Robin solver..."*) forces the stencil to evaluate `mlp_m_fn` at ghost nodes in $\Omega^+$. Because there are no explicit equations constraining `mlp_m_fn` in $\Omega^+$ (the loss is only computed for $V_i > 0$), the neural network's weights are underdetermined. It can exploit these "free" ghost nodes to perfectly force the discrete residuals to 0 without forcing the interior nodes to accurately match the continuous PDE. As $N$ increases to 16, the number of ghost cells increases, amplifying this underdetermination effect.

### How to achieve true 2nd-Order Convergence:
To fix this and get $O(\Delta x^2)$ linear descent in the log-scale, you must upgrade the boundary term to second-order. You can do this by extrapolating $u$ from the cell center to the interface using a Taylor expansion:
$$ u(x_\Gamma) \approx u_{ijk} + \nabla u_{ijk} \cdot (x_\Gamma - x_i) $$
Which turns the integral into:
$$ \int_\Gamma \alpha(x) u(x) dS \approx u_{ijk} \int_\Gamma \alpha dS + \nabla u_{ijk} \cdot \int_\Gamma \alpha(x) (x - x_i) dS $$

Since you already use central differences for $\nabla u_{ijk}$ in the flux, adding this correction vector term to your `lhs` stencil will eliminate the $O(1)$ truncation error and immediately restore 2nd-order accuracy without needing to change your MLP architecture!
