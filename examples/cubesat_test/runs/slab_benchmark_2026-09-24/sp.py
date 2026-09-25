import h5py, numpy as np, sys
NA=6.02214076e23
LIB='/Users/massimolarsen/Documents/Repos/MCDC/hdf5lib'
AMASS={'Al':26.9815,'Si':28.0855,'Cu':63.546,'O':15.999,'C':12.011,'H':1.008}
def interp(x,xp,fp):
    # code: linear interp, find_bin
    return np.interp(x,xp,fp)

def mean_min(v,c,cap):
    # E[min(T,cap)] for linear CDF: integral_0^cap (1-F)
    c=c/c[-1]
    # 1-F piecewise linear on v; below v[0]: F=0
    t=[0.0]+list(v); F=[0.0]+list(c)
    t=np.array(t);F=np.array(F)
    if cap<=t[-1]:
        Fc=np.interp(cap,t,F); m=t<cap
        tt=np.append(t[m],cap); FF=np.append(F[m],Fc)
    else:
        tt=np.append(t,cap);FF=np.append(F,1.0)
    return np.trapezoid(1-FF,tt)

class El:
    def __init__(s,name):
        f=h5py.File(f'{LIB}/{name}.h5','r'); r=f['electron_reactions']
        s.E=r['xs_energy_grid'][()]*1e6
        s.el=r['elastic_scattering/MT-526/xs'][()]
        s.lg=r['elastic_scattering/MT-526/large_angle/xs'][()]
        s.tr=r['elastic_scattering/MT-526/large_angle/transport'][()]
        s.exc=r['excitation/MT-528/xs'][()]
        s.brem=r['bremsstrahlung/MT-527/xs'][()]
        s.ion=r['ionization/MT-522/xs'][()]
        x=r['excitation/MT-528/energy_loss']; s.excE=x['energy'][()]*1e6; s.excV=x['value'][()]*1e6
        x=r['bremsstrahlung/MT-527/energy_loss']; s.brE=x['energy'][()]*1e6; s.brV=x['value'][()]*1e6
        s.sub=[]
        for k in r['ionization/MT-522/subshells']:
            g=r['ionization/MT-522/subshells/'+k]
            p=g['product']
            o=list(p['energy_offset'][()])+[len(p['value'])]
            v=p['value'][()]*1e6; c=p['CDF'][()]
            tabs=[(v[o[i]:o[i+1]],c[o[i]:o[i+1]]) for i in range(len(o)-1)]
            s.sub.append(dict(xsE=g['energy_grid'][()]*1e6,xs=g['xs'][()],B=g['binding_energy'][()]*1e6,
                              grid=p['energy_grid'][()]*1e6,tabs=tabs))
        cos=r['elastic_scattering/MT-526/large_angle/scattering_cosine']
        s.cg=cos['energy_grid'][()]*1e6; o=list(cos['energy_offset'][()])+[len(cos['value'])]
        s.ctabs=[(cos['value'][()][o[i]:o[i+1]],cos['CDF'][()][o[i]:o[i+1]]) for i in range(len(o)-1)]
        f.close()
    def T_mean(s,E,sh,mode='code'):
        g=sh['grid']; B=sh['B']
        if E<g[0]: return [(1.0,0)]
        if E>=g[-1]: return [(1.0,len(g)-1)]
        i=np.searchsorted(g,E,side='right')-1
        f=(E-g[i])/(g[i+1]-g[i])
        return [(1-f,i),(f,i+1)]
    def loss_ion(s,E,sh,mode='code'):
        # mean energy lost by primary per ionization in subshell (B+T, capped at E), and mean T
        B=sh['B']
        if E<=B: return E,0
        L=0; Tm=0
        if mode=='code':
            for w,i in s.T_mean(E,sh):
                v,c=sh['tabs'][i]
                L+=w*(B+mean_min(v,c,E-B)); Tm+=w*mean_min(v,c,1e30)
        else:
            # scaled / unit-base: interpolate in reduced variable T/((E-B)/2) using log-E weight
            g=sh['grid']
            if E>=g[-1]: i=len(g)-2
            else: i=max(0,np.searchsorted(g,E,side='right')-1)
            f=np.log(E/g[i])/np.log(g[i+1]/g[i])
            for w,j in [(1-f,i),(f,i+1)]:
                v,c=sh['tabs'][j]; Tmax_j=v[-1]; Tmax=(E-B)/2
                vs=v/Tmax_j*Tmax
                m=mean_min(vs,c,1e30); L+=w*(B+m); Tm+=w*m
        return L,Tm
    def at(s,E,mode='code'):
        el=interp(E,s.E,s.el); exc=interp(E,s.E,s.exc); br=interp(E,s.E,s.brem); ion=interp(E,s.E,s.ion)
        dexc=interp(E,s.excE,s.excV); dbr=interp(E,s.brE,s.brV)
        ionL=0; sumsub=0
        for sh in s.sub:
            xs=interp(E,sh['xsE'],sh['xs']); sumsub+=xs
            L,_=s.loss_ion(E,sh,mode); ionL+=xs*L
        return dict(el=el,exc=exc,br=br,ion=ion,sumsub=sumsub,dexc=dexc,dbr=dbr,
                    Scol_b=exc*dexc+ionL, Srad_b=br*dbr, tot=el+exc+br+ion)
