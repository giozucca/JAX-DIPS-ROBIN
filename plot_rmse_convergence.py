import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def calculate_rmse(npz_path):
    data = np.load(npz_path)
    # The error array U_error is actually U - U_exact
    u_error = data['U_error']
    phi = data['phi']
    
    # Calculate RMSE over the entire domain
    rmse_total = np.sqrt(np.mean(u_error**2))
    
    # Calculate RMSE inside the domain (phi < 0)
    inside_domain = phi < 0
    if np.any(inside_domain):
        rmse_inside = np.sqrt(np.mean(u_error[inside_domain]**2))
    else:
        rmse_inside = rmse_total
        
    return rmse_total, rmse_inside

def main():
    resolutions = [8, 16, 32, 64]
    dirs = [f"star_Robin3_{r}" for r in resolutions]
    
    rmse_totals = []
    rmse_insides = []
    valid_res = []
    
    for r, d in zip(resolutions, dirs):
        npz_file = os.path.join(d, f"{d}.npz")
        if os.path.exists(npz_file):
            rt, ri = calculate_rmse(npz_file)
            rmse_totals.append(rt)
            rmse_insides.append(ri)
            valid_res.append(r)
            print(f"Resolution {r}: RMSE (total) = {rt:.6e}, RMSE (inside phi<0) = {ri:.6e}")
        else:
            print(f"File not found: {npz_file}")
            
    if not valid_res:
        print("No valid data found to plot.")
        return

    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot RMSE inside the domain
    plt.loglog(valid_res, rmse_insides, marker='o', linestyle='-', linewidth=2, markersize=8, label='RMSE (inside domain, $\phi < 0$)')
    
    # Plot RMSE over total domain
    plt.loglog(valid_res, rmse_totals, marker='s', linestyle='--', linewidth=2, markersize=8, label='RMSE (total domain)')
    
    # Reference slopes for convergence rates
    # e.g., O(h) and O(h^2) lines. We'll use the last valid_res point to anchor them.
    N_ref = np.array(valid_res)
    h_ref = 1.0 / N_ref
    
    # Anchor the reference lines to the first data point
    base_error = rmse_insides[0]
    first_h = h_ref[0]
    
    # O(h^1) reference
    ref_1 = base_error * (h_ref / first_h)**1
    # O(h^2) reference
    ref_2 = base_error * (h_ref / first_h)**2
    
    plt.loglog(valid_res, ref_1, 'k:', label='$\mathcal{O}(h)$ reference')
    plt.loglog(valid_res, ref_2, 'k-.', label='$\mathcal{O}(h^2)$ reference')

    plt.xlabel('Resolution (Nx)', fontsize=14)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=14)
    plt.title('RMSE vs System Resolution (star_Robin3)', fontsize=16)
    
    # Set x-ticks explicitly
    plt.xticks(valid_res, labels=[str(r) for r in valid_res])
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    output_pdf = "rmse_convergence_star_Robin3.pdf"
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', dpi=300)
    plt.savefig("rmse_convergence_star_Robin3.png", format='png', bbox_inches='tight', dpi=300)
    print(f"\nPlot saved as {output_pdf} and rmse_convergence_star_Robin3.png")

if __name__ == "__main__":
    main()
