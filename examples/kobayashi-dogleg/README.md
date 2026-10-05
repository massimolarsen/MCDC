# Kobayashi dog-leg examples

- `steady_state`: original steady-state shielding benchmark from [Kobayashi et al. (2001)](https://doi.org/10.1016/S0149-1970%2801%2900007-5).
- `pulsed`: adaptation of the [time-dependent benchmark on Zenodo](https://doi.org/10.5281/zenodo.15069882), with a 0–50 s source pulse, an outlet detector, and tallies through 200 s.
- `pulsed_with_fission`: pulsed source with a fuel cube at the second turn, an outlet detector, and tallies through 500 s, emphasizing the later fission contributions.

Each folder contains `input.py` and `process-output.py`.
Run from that folder:

```sh
python input.py
python process-output.py
```

Both pulsed examples use a 10 cm detector cube at x=30–40, y=90–100, z=30–40 cm, filled with shield material (capture and scattering each 0.05 cm⁻¹).
Their processors show a flux map above density and detector response curves, with uncertainty bands and moving time markers.
Detector capture uses 1 s bins; flux and density use 10 s bins.
