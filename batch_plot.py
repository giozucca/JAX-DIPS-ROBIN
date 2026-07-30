import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def plot_file(filepath):
    print(f"Loading {os.path.basename(filepath)}...")
    data = np.load(filepath)
    X = data['X']
    Y = data['Y']
    Z = data['Z']
    phi = data['phi']
    U = data['U']
    U_exact = data['U_exact']
    U_error = data['U_error']
    
    nx, ny, nz = X.shape
    axis_choice = 'Z'
    slice_idx = nz // 2
    
    x_coords = X[:, :, slice_idx]
    y_coords = Y[:, :, slice_idx]
    phi_slice = phi[:, :, slice_idx]
    u_slice = U[:, :, slice_idx]
    exact_slice = U_exact[:, :, slice_idx]
    err_slice = U_error[:, :, slice_idx]
    xlabel, ylabel = "X", "Y"

    filename_lower = os.path.basename(filepath).lower()
    default_err_max = 0.025 if "sphere" in filename_lower else 0.5
    use_fixed = True

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Simulation Slice: {os.path.basename(filepath)}\nSlice along {axis_choice} = {slice_idx}", fontsize=16)

    def plot_field(ax, x, y, field, title, cmap='viridis', is_error=False, mask_outside=True, vmin=None, vmax=None):
        if is_error:
            field_plot = np.abs(field)
            title = title + " (Abs)"
        else:
            field_plot = field
            
        if mask_outside:
            field_plot = np.where(phi_slice >= 0, np.nan, field_plot)
            
        im = ax.pcolormesh(x, y, field_plot, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
        fig.colorbar(im, ax=ax)
        
        if np.min(phi_slice) < 0 < np.max(phi_slice):
            ax.contour(x, y, phi_slice, levels=[0.0], colors='red', linewidths=2)
            
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_aspect('equal')

    plot_field(
        axs[0, 0], x_coords, y_coords, phi_slice, 
        "Level Set (phi)\n[Red Line shows Interface phi=0]", 
        cmap='bwr', mask_outside=False,
        vmin=-2.0 if use_fixed else None,
        vmax=2.0 if use_fixed else None
    )
    plot_field(
        axs[0, 1], x_coords, y_coords, u_slice, 
        "Numerical Solution (U)", 
        mask_outside=True,
        vmin=-1.0 if use_fixed else None,
        vmax=1.0 if use_fixed else None
    )
    plot_field(
        axs[1, 0], x_coords, y_coords, exact_slice, 
        "Exact Solution (U_exact)", 
        mask_outside=True,
        vmin=-1.0 if use_fixed else None,
        vmax=1.0 if use_fixed else None
    )
    plot_field(
        axs[1, 1], x_coords, y_coords, err_slice, 
        "Absolute Error |U - U_exact|", 
        cmap='inferno', is_error=True, mask_outside=True,
        vmin=0.0 if use_fixed else None,
        vmax=default_err_max if use_fixed else None
    )

    plt.tight_layout()
    
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    pdf_filename = f"{base_name}_{axis_choice}_slice{slice_idx}_2x2.pdf"
    pdf_path = os.path.join(os.path.dirname(filepath), pdf_filename)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved vector PDF to: {pdf_path}")
    plt.close()

if __name__ == "__main__":
    dirs = ["star_Robin3_8", "star_Robin3_16", "star_Robin3_32", "star_Robin3_64"]
    for d in dirs:
        npz_files = glob.glob(os.path.join(d, "*.npz"))
        for npz in npz_files:
            plot_file(npz)
