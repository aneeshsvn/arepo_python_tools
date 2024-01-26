import numpy as np
from global_props import get_particle_data
from com import mediancenter

def radial_avg(path,snap,ptype,prop,rmin,rmax,bins):
    pdata = get_particle_data(path,snap,ptype,['Coordinates',prop])
    center = mediancenter(path,snap)
    x = pdata['Coordinates']-center
    x = np.linalg.norm(x, axis=1)
    mask = (x >= rmin) & (x <= rmax)
    x = x[mask]
    part_prop = pdata[prop][mask]
    dr = (rmax-rmin)/bins
    avg = np.zeros(bins)
    freq = np.zeros(bins)
    for i in range(len(x)):
        idx = int((x-rmin)/dr)
        avg[idx] += part_prop[i]
        freq[idx] += 1
    avg /= freq
    edge = np.linspace(rmin,rmax,bins+1)
    return 0.5*(edge[:-1]+edge[1:]), avg  

