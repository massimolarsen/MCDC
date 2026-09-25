"""Summarize MC/DC vs Geant4 slab benchmark into results.md / results.csv."""
import glob, re, numpy as np, h5py
cases = [(0.1, 0.3), (0.255, 0.3), (1.0, 0.3), (4.0, 0.3), (6.0, 0.3), (10.0, 0.3), (0.1, 0.003), (0.255, 0.012)]
edges = [0, 1e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 12]  # MeV, same in both codes
g4 = {}
for line in open('g4_results.txt'):
    if line.startswith('RESULT '):
        d = dict(kv.split('=') for kv in line.split()[1:]); key = (float(d['E_MeV']), float(d['T_cm']))
        g4[key] = {k: float(v) for k, v in d.items()}
    elif line.startswith('RESULT_BINS'):
        g4[key]['bins'] = np.array(line.split()[1:], float)
def ci(k, n):  # 1-sigma Poisson on a count per source electron (>=1 count floor)
    return np.sqrt(max(k, 1.0)) / n
rows = []
for E, T in cases:
    tag = f"E{E:g}MeV_T{T:g}"
    fn = f"mcdc_{tag}.h5"
    try:
        f = h5py.File(fn, 'r')
    except OSError:
        rows.append((E, T, None, None, g4.get((E, T)))); continue
    N = int(f['settings/N_particle'][()])
    det = f['tallies/detector/current-in/mean'][()].reshape(-1)          # per source electron, per energy bin
    slab = f['tallies/slab/current-out/mean'][()]
    slab = slab.reshape(slab.shape[0], -1, 6)[:, 0, :] if slab.ndim > 2 else slab
    back = float(np.sum(f['tallies/slab/current-out/mean'][()].reshape(len(edges) - 1, 6, -1)[:, 0, :]))
    trans = float(np.sum(f['tallies/slab/current-out/mean'][()].reshape(len(edges) - 1, 6, -1)[:, 1, :]))
    rows.append((E, T, dict(N=N, det=det.sum(), det50=det[3:].sum(), back=back, trans=trans), det, g4.get((E, T))))
out = ["| E (MeV) | Al (cm) | MC/DC N | MC/DC det. entries / e⁻ | Geant4 det. entries / e⁻ | MC/DC ÷ G4 | MC/DC backscatter | G4 backscatter |",
       "|---|---|---|---|---|---|---|---|"]
csv = ["E_MeV,T_cm,mcdc_N,mcdc_det_per_e,mcdc_det_sigma,g4_N,g4_det_per_e,g4_det_sigma,mcdc_back,g4_back,mcdc_trans,g4_trans"]
for r in rows:
    E, T, m = r[0], r[1], r[2]; g = r[-1]
    gN = g['N']; gd = g['det_all'] / gN; gs = ci(g['det_all'], gN); gb = g['slab_back'] / gN; gt = g['slab_trans'] / gN
    if m is None:
        out.append(f"| {E:g} | {T:g} | – | (not run) | {gd:.4f} ± {gs:.4f} | – | – | {gb:.3f} |"); continue
    ms = ci(m['det'] * m['N'], m['N'])
    ratio = f"{m['det']/gd:.2f} ± {m['det']/gd*np.hypot(ms/max(m['det'],1e-12), gs/gd):.2f}" if gd > 0 else ("–" if m['det'] == 0 else "∞")
    out.append(f"| {E:g} | {T:g} | {m['N']} | {m['det']:.4f} ± {ms:.4f} | {gd:.4f} ± {gs:.4f} | {ratio} | {m['back']:.3f} ± {ci(m['back']*m['N'], m['N']):.3f} | {gb:.3f} ± {ci(g['slab_back'], gN):.3f} |")
    csv.append(f"{E},{T},{m['N']},{m['det']:.6f},{ms:.6f},{int(gN)},{gd:.6f},{gs:.6f},{m['back']:.6f},{gb:.6f},{m['trans']:.6f},{gt:.6f}")
open('results.md', 'w').write("\n".join(out) + "\n"); open('results.csv', 'w').write("\n".join(csv) + "\n")
print("\n".join(out))
