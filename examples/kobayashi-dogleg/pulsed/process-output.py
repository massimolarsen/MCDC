import numpy as np
import matplotlib.pyplot as plt
import h5py
import matplotlib.animation as animation
import sys

# Load result
with h5py.File(sys.argv[1] if len(sys.argv) > 1 else "output.h5", "r") as f:
    x = f["tallies/mesh_flux/grid/x"][:]
    x_mid = 0.5 * (x[:-1] + x[1:])
    y = f["tallies/mesh_flux/grid/y"][:]
    y_mid = 0.5 * (y[:-1] + y[1:])
    t = f["tallies/mesh_flux/grid/time"][:]
    t_mid = 0.5 * (t[:-1] + t[1:])
    X, Y = np.meshgrid(y, x)

    phi = f["tallies/mesh_flux/flux/mean"][:]
    phi_sd = f["tallies/mesh_flux/flux/sdev"][:]

    phi_total = f["tallies/density/density/mean"][:]
    phi_total_sd = f["tallies/density/density/sdev"][:]
    capture = f["tallies/detector/capture/mean"][:]
    capture_sd = f["tallies/detector/capture/sdev"][:]
    t_detector = f["tallies/detector/grid/time"][:]
    t_detector_mid = 0.5 * (t_detector[:-1] + t_detector[1:])

# Animate result
fig = plt.figure(figsize=(6, 6), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
ax_flux = fig.add_subplot(grid[0, :])
ax_density = fig.add_subplot(grid[1, 0])
ax_detector = fig.add_subplot(grid[1, 1])
#
cax = ax_flux.pcolormesh(X, Y, phi[0], vmin=phi[0].min(), vmax=phi[0].max())
ax_flux.set_aspect("equal", "box")
ax_flux.set_xlabel("$y$ [cm]")
ax_flux.set_ylabel("$x$ [cm]")
#
ax_density.set_xlabel("$t$ [s]")
ax_density.set_title("Neutron density")
ax_density.set_yscale("log")
ax_density.plot(t_mid, phi_total, "b-")
ax_density.fill_between(
    t_mid, phi_total - phi_total_sd, phi_total + phi_total_sd, alpha=0.2, color="b"
)
ax_density.grid()
ax_density.set_box_aspect(0.65)
(density_marker,) = ax_density.plot([], [], "ok", fillstyle="none")

# Cell tallies are integrated over each time bin; divide by its width for rates.
capture_rate = capture / np.diff(t_detector)
capture_rate_sd = capture_sd / np.diff(t_detector)
(detector_line,) = ax_detector.plot(t_detector_mid, capture_rate, "g-")
ax_detector.fill_between(
    t_detector_mid,
    capture_rate - capture_rate_sd,
    capture_rate + capture_rate_sd,
    alpha=0.2,
    color=detector_line.get_color(),
)
ax_detector.set_xlabel("$t$ [s]")
ax_detector.set_title("Detector capture rate")
ax_detector.set_yscale("log")
ax_detector.grid()
ax_detector.set_box_aspect(0.65)
(detector_marker,) = ax_detector.plot([], [], "ok", fillstyle="none")


#
def animate(i):
    density_marker.set_data([t_mid[i]], [phi_total[i]])
    # Follow the detector's finer time grid at the current flux-frame time.
    detector_value = np.interp(t_mid[i], t_detector_mid, capture_rate)
    detector_marker.set_data([t_mid[i]], [detector_value])
    ax_flux.set_title(f"Flux map: $t=$ {t[i]:g}–{t[i + 1]:g} s")
    cax.set_array(phi[i])
    cax.set_clim(phi[i].min(), phi[i].max())


#
K = len(t) - 1
animate(0)
anim = animation.FuncAnimation(fig, animate, frames=K)
plt.show()
