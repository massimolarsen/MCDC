.. _example_cubesat:

=============================================
CubeSat Sensitive-Volume Energy Deposition
=============================================

Problem Description
===================

A preliminary continuous-energy CubeSat-like fixed-source problem built
from rectangular constructive solid geometry.  The model represents a
1U CubeSat envelope with aluminum rails, shear panels, solar panels,
board stack, antenna strips, and four electronics-sensitive volumes.

This example provides a bare-bones starting point for CARRE-driven
workflow development by scoring energy deposition in representative
sensitive volumes.

Geometry and Materials
======================

The CubeSat footprint spans :math:`x,y \in [0,10]` cm, with large rails
covering :math:`z \in [0,11]` cm.  A 1 m vacuum boundary cube surrounds
the model and is centered on the CubeSat.

The geometry includes:

* four Al7075 corner rails;
* eight Al7075 edge rails;
* six Al6061 shear panels;
* ten silicon solar panels;
* four copper antenna strips;
* four FR4-proxy circuit boards;
* four representative sensitive volumes for OBC, EPS, ADCS, and Comms.

Continuous-energy material compositions are specified as nuclide atom
densities in atoms/barn-cm.

Physical Assumptions
====================

* Continuous-energy neutron transport using the configured HDF5 nuclear
  data library.
* Monoenergetic 14 MeV source particles.
* Source particles are sampled uniformly from the six faces of the
  surrounding boundary cube and emitted into the inward-facing
  hemisphere.
* Sensitive-volume energy deposition is used as an initial proxy
  quantity for downstream electronics-effect analysis.

Numerical Setup
===============

.. list-table::
   :widths: 35 65

   * - **Tally type**
     - Cell-filtered collision tallies
   * - **Tally score**
     - Energy deposition in the OBC, EPS, ADCS, and Comms sensitive volumes
   * - **Tally energy grid**
     - Log-spaced bins from 10 eV to 20 MeV, with a 0 eV lower edge
   * - **Source particles**
     - :math:`10^{5}` (demonstration)
   * - **Output file**
     - ``cubesat_CE.h5``

Quantities of Interest
======================

* Energy deposited in each representative sensitive volume as a
  function of particle energy.
* Statistical uncertainty for each energy-deposition tally.

Limitations and Follow-up Needs
===============================

This example is intentionally bare-bones.  The geometry and material
definitions are representative rather than a validated spacecraft model,
the source is a simple monoenergetic inward-biased boundary source, and
the default particle count is chosen for a quick demonstration rather
than production statistics.

Future CARRE-driven refinements can add higher-statistics runs,
application-specific spectra, Geant4 coupling workflows, and
post-processing that maps deposited energy to SEE or related
electronics-effect metrics.

Step-by-Step Walkthrough
========================

**1. Imports and Model Overview (lines 1-16)**

.. literalinclude:: ../../../examples/cubesat/input.py
   :language: python
   :lines: 1-16
   :linenos:
   :lineno-match:

The input imports NumPy and MC/DC, defines the tally energy grid, then
summarizes the CubeSat geometry and continuous-energy material
convention used by the example.

**2. Materials and Box Helper (lines 18-100)**

.. literalinclude:: ../../../examples/cubesat/input.py
   :language: python
   :lines: 18-100
   :linenos:
   :lineno-match:

Materials are defined from nuclide atom densities.  The ``box`` helper
creates axis-aligned rectangular CSG regions from six planes.

**3. Geometry and Sensitive Volumes (lines 103-263)**

.. literalinclude:: ../../../examples/cubesat/input.py
   :language: python
   :lines: 103-263
   :linenos:
   :lineno-match:

The CubeSat body is assembled from rails, panels, solar cells, antenna
strips, and a board stack.  Four sensitive-volume cells are retained for
electronics-effect proxy scoring.

**4. Source (lines 265-312)**

.. literalinclude:: ../../../examples/cubesat/input.py
   :language: python
   :lines: 265-312
   :linenos:
   :lineno-match:

Six equal-probability sources sample the faces of the vacuum boundary
cube.  The source points are placed just inside the boundary to avoid
starting directly on a vacuum surface, and each source samples the
hemisphere directed toward the CubeSat.

**5. Tallies, Settings, and Run (lines 314-end)**

.. literalinclude:: ../../../examples/cubesat/input.py
   :language: python
   :lines: 314-
   :linenos:
   :lineno-match:

Each sensitive volume has a cell-filtered ``energy_deposition`` tally
with energy bins.  The example writes results to ``cubesat_CE.h5``.

**What to try:**

* Increase ``mcdc.settings.N_particle`` for lower statistical
  uncertainty.
* Change ``source_energy_ev`` or replace the monoenergetic source with a
  mission-specific spectrum.
* Refine ``ENERGY_BINS_EV`` or add additional sensitive-volume cells for
  subsystem studies.

Full Input
==========

Click here to view the input file: `examples/cubesat/input.py <https://github.com/CEMeNT-PSAAP/MCDC/blob/dev/examples/cubesat/input.py>`_.

The complete input used for this example is embedded below:

.. literalinclude:: ../../../examples/cubesat/input.py
   :language: python
   :linenos:

How to Run
==========

From the repository root run::

  python examples/cubesat/input.py
  python examples/cubesat/process-output.py

Expected Output
===============

The input writes ``cubesat_CE.h5``.  The companion
``process-output.py`` script prints a table of mean energy deposition
and standard deviation, in eV per source particle, for each
sensitive-volume tally.  It also saves a combined energy-deposition
curve plot in the ``cubesat_plots`` directory.
