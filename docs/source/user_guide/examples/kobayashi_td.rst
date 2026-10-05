.. _example_kobayashi_td:

=============================================
Pulsed Kobayashi Dog-Leg with Fission
=============================================

Description and Reference Problems
==================================

The original steady-state shielding problem comes from Kobayashi, Sugimura, and Nagaya's *Progress in Nuclear Energy* paper [KobayashiPulsed2001]_.
It supplies the dog-leg geometry and the non-fissioning shielding problem described in :ref:`example_kobayashi_dog_leg`.
Variansyah's Zenodo record [Variansyah2025]_ adapts that problem to a pulsed source and provides a time-dependent benchmark with archived OpenMC calculations.
The PNE paper is the reference for the original steady-state problem, while the Zenodo adaptation is the reference for the pulsed formulation.

This walkthrough uses ``examples/kobayashi-dogleg/pulsed_with_fission``.
It extends the pulsed problem with a fuel cube at the second turn, a material detector at the outlet, and tallies through 500 s.
These material changes define an example derived from the references; their detector response is not a reference solution published in the original paper or the Zenodo benchmark.
The sibling ``pulsed`` example includes the outlet detector without the fuel cube and retains a 200 s tally window.
For comparisons with the archived non-fissioning transient benchmark, see :ref:`project_vvp_code_to_code_neutron_kobayashi`.

Step-by-Step Walkthrough
========================

**1. Fuel and Detector**

The domain and outer boundary conditions follow the steady-state dog-leg example.
A 10 cm fuel cube occupies x=30–40, y=50–60, z=0–10 cm, at the turn from the x-directed channel into the z-directed channel.
It has scattering and fission cross sections of 0.05 cm\ :sup:`-1` each, zero capture, and a prompt fission multiplicity of 2.5.

.. literalinclude:: ../../../../examples/kobayashi-dogleg/pulsed_with_fission/input.py
   :language: python
   :start-at: m_fuel =
   :end-at: )
   :linenos:
   :lineno-match:

The detector occupies x=30–40, y=90–100, z=30–40 cm at the outlet.
It uses the shield material, with capture and scattering cross sections of 0.05 cm\ :sup:`-1` each.
Both cubes replace portions of the low-density channel.

**2. Source with a Time Window**

.. literalinclude:: ../../../../examples/kobayashi-dogleg/pulsed_with_fission/input.py
   :language: python
   :start-at: source = mcdc.Source(
   :end-at: simulation.set_sources([source])
   :linenos:
   :lineno-match:

The source emits isotropically from the original corner cube over ``time=[0.0, 50.0]``.
This finite pulse introduces the time dependence inherited from the Zenodo adaptation.

**3. Time-Resolved Tallies**

.. literalinclude:: ../../../../examples/kobayashi-dogleg/pulsed_with_fission/input.py
   :language: python
   :start-at: time_grid =
   :end-at: simulation.set_tallies(
   :linenos:
   :lineno-match:

The mesh flux and global density tallies use 10 s bins through 500 s.
The global density tally records neutron population integrated over each time bin.
Detector capture uses a finer 1 s grid to resolve the response.
The input uses 100,000 source histories per batch and two batches, with implicit capture enabled.

**What to try:**

- Shorten the source pulse to resolve arrival times more clearly.
- Increase the number of histories and batches to improve detector statistics.
- Compare with the sibling ``pulsed`` example over their common time interval, accounting for their different material at the second turn.

Full Input
==========

View the input file: `examples/kobayashi-dogleg/pulsed_with_fission/input.py <https://github.com/mcdc-project/mcdc/blob/dev/examples/kobayashi-dogleg/pulsed_with_fission/input.py>`_.

.. literalinclude:: ../../../../examples/kobayashi-dogleg/pulsed_with_fission/input.py
   :language: python
   :linenos:

How to Run
==========

From inside ``examples/kobayashi-dogleg/pulsed_with_fission`` run::

   python input.py
   python process-output.py

Expected Output
===============

The HDF5 output includes mesh flux, global density, and detector capture.
``process-output.py`` places the flux map above two shorter time-response plots, with uncertainty bands and synchronized moving circles.
Detector capture and its standard deviation are divided by each detector time-bin width to show a rate per source neutron.
The animation uses the coarser flux time grid and interpolates the detector curve for its moving marker.

References
==========

.. [KobayashiPulsed2001] K. Kobayashi, N. Sugimura, and Y. Nagaya, `3D Radiation Transport Benchmark Problems and Results for Simple Geometries with Void Region <https://doi.org/10.1016/S0149-1970(01)00007-5>`_, *Progress in Nuclear Energy*, **39** (2), 119–144 (2001).

.. [Variansyah2025] I. Variansyah, `Time-Dependent Kobayashi Dog-Leg Benchmark for Neutron Transport <https://doi.org/10.5281/zenodo.15069882>`_, Zenodo (2025).
