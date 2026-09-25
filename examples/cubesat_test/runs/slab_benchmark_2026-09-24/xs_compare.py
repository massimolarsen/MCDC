"""Compare MC/DC (EPRDATA14 data, as sampled by MC/DC) vs Geant4 option4 physics for the benchmark materials."""
import sys, numpy as np
from sp import El
alpha=7.2973525693e-3; mc2=0.51099895e6; xc=1e-6
def eta(E,Z):  # native.py compute_scattering_eta, E in eV
    pc=np.sqrt(E*(E+2*mc2)); beta=pc/(E+mc2); tau=E/mc2; r=alpha*mc2/(0.885*pc)
    return 0.25*r*r*Z**(2/3)*(1.13+3.76*(alpha*Z/beta)**2)*np.sqrt(tau/(tau+1))
def msa(e): return (np.log((e+xc)/e)-xc/(e+xc))/(xc/(e*(e+xc)))
ZZ={'Al':13,'Mg':12,'Si':14}
mats={'Al6061':{'Al':0.059057360465045,'Mg':0.00081349602416576,'Si':0.00046494828605733},
      'Silicon':{'Si':0.050132618436459}}
els={k:El(k) for k in ZZ}
for el in els.values():
    el.m=np.array([1-np.sum(0.5*(v[1:]+v[:-1])*np.diff(c)) for v,c in el.ctabs])
g4={}
for line in open('g4_calc.txt'):
    if line.startswith('CALC'):
        _,mn,e,sc,sr,xm=line.split(); g4.setdefault(mn,[]).append([float(e),float(sc),float(sr),float(xm)])
out=[]
for mn,comp in mats.items():
    G=np.array(g4[mn]); lg=lambda E,col: np.exp(np.interp(np.log(E),np.log(G[:,0]),np.log(np.maximum(G[:,col],1e-300))))
    out.append(f'\n{mn}   (ratios MC/DC / Geant4-option4)')
    out.append(' E(MeV)  S_col  S_rad  S_tot | Sigma_tr(file data)  Sigma_tr(MC/DC as sampled)')
    for E in [0.01,0.05,0.1,0.255,0.5,1,2,4,6,10]:
        Ee=E*1e6; sc=sr=trf=trc=0
        for s,N in comp.items():
            d=els[s].at(Ee); el=els[s]
            sc+=N*d['Scol_b']/1e6; sr+=N*d['Srad_b']/1e6
            slg=np.interp(Ee,el.E,el.lg); sel=np.interp(Ee,el.E,el.el)
            trf+=N*np.interp(Ee,el.E,el.tr)
            trc+=N*(slg*np.interp(Ee,el.cg,el.m)+(sel-slg)*msa(eta(Ee,ZZ[s])))
        gsr,gx=lg(E,2),lg(E,3); gsc=lg(E,1)-gsr  # G4 ElectronicDEDX includes eBrem
        out.append(f' {E:6.3f}  {sc/gsc:5.2f}  {sr/gsr:5.2f}  {(sc+sr)/(gsc+gsr):5.2f} |      {trf/gx:5.2f}                 {trc/gx:6.2f}')
print('\n'.join(out))
