import numpy as np
import illustris_python as il
from global_props import get_particle_data
from com import get_iter_com, mediancenter
import multiprocessing
from multiprocessing import Pool
import glob
# import matplotlib.pyplot as plt


def weighted_median(output_path, snapnum):
    # returns the weighted median of the particle positions
    masses=np.array([])
    pos=np.array([]).reshape(0,3)
    for ptype in range(6):
        particles = il.snapshot.loadSubset(output_path,snapnum,ptype,fields=['Coordinates'],sq=False)
        if particles['count'] > 0:
            m=il.snapshot.loadHeader(output_path,snapnum).get('MassTable')[ptype]
            if m == 0:
                mm = il.snapshot.loadSubset(output_path,snapnum,ptype,fields=['Masses'])
            else:
                mm=np.array([m]*particles['count'])
            masses = np.concatenate((masses,mm))
            pos = np.concatenate((pos,particles['Coordinates']),axis=0)    
    
    pairs = []
    
    W = masses/np.sum(masses)
    wmed=[]
    
    for i in range(3):
        x = pos[:,i]
        mask = np.argsort(x)
        x=x[mask]
        ww=W[mask]
        wsum = 0
        for j in range(len(x)):
            wsum += ww[j]
            if wsum >= 0.5:
                wmed.append(x[j])
                break
    return {'x':wmed[0],'y':wmed[1],'z':wmed[2]}
    
    
def deltar(dat1,dat2):
    ''' dat1 is a dictionary with keys 'x','y','z'
     dat2 can be a float, a list, or a dictionary with keys 'x','y','z'
     returns the distance between dat1 and dat2'''
    if isinstance(dat2, (float,int)):
        return np.sqrt((dat1['x']-dat2)**2 + (dat1['y']-dat2)**2 + (dat1['z']-dat2)**2)
    elif isinstance(dat2, (list,np.ndarray)) & len(dat2) == 3:
        return np.sqrt((dat1['x']-dat2[0])**2 + (dat1['y']-dat2[1])**2 + (dat1['z']-dat2[2])**2)
    elif isinstance(dat2, dict):
        return np.sqrt((dat1['x']-dat2['x'])**2 + (dat1['y']-dat2['y'])**2 + (dat1['z']-dat2['z'])**2)

def rhalfstar(path,snap,center='bh',R=10):
    stars = get_particle_data(path,snap,'234',['Coordinates','Masses'])
    # center = get_iter_com(path,snap,guess=center,R=R)
    center = mediancenter(path,snap)
    r = np.linalg.norm(stars['Coordinates']-center,axis=1)
    ids = np.argsort(r)
    sorted_r = r[ids]
    masses = stars['Masses'][ids]
    cummass = np.cumsum(masses)
    for i in range(len(r)):
        if cummass[i] >= cummass[-1]/2:
            return sorted_r[i]
        
def scaleheight(path,snap,center):
    gas = il.snapshot.loadSubset(path,snap,0,['Coordinates','Masses','Velocities'])
    cent = get_iter_com(path,snap,center,10)
    pos = gas['Coordinates'] - cent
    r=np.linalg.norm(pos,axis=1)
    R=5
    pos=pos[r<R]
    masses=gas['Masses'][r<R]
    vel=gas['Velocities'][r<R]
    totL=np.array([np.sum(masses*(pos[:,1]*vel[:,2]-pos[:,2]*vel[:,1])),
                   np.sum(masses*(pos[:,2]*vel[:,0]-pos[:,0]*vel[:,2])),
                   np.sum(masses*(pos[:,0]*vel[:,1]-pos[:,1]*vel[:,0]))])
    zaxis = totL/np.linalg.norm(totL)

    ct=zaxis[2]/np.sqrt(zaxis[0]**2+zaxis[1]**2+zaxis[2]**2)
    st=np.sqrt(1-ct**2)
    cp=zaxis[0]/np.sqrt(zaxis[0]**2+zaxis[1]**2)
    sp=zaxis[1]/np.sqrt(zaxis[0]**2+zaxis[1]**2)
    
    x1 = gas['Coordinates'][:,0] - cent[0]
    y1 = gas['Coordinates'][:,1] - cent[1]
    z1 = gas['Coordinates'][:,2] - cent[2]

    xpos=x1*ct*cp+y1*sp*ct-st*z1
    ypos=-x1*sp+y1*cp
    zpos=x1*st*cp+y1*st*sp+z1*ct
    
    mask = (xpos**2 + ypos**2 < 3**2) & (np.abs(zpos)<3)
    masses = gas['Masses'][mask]
    zpos = zpos[mask]
    zmin = -3.0; zmax = 3.0; dz = 0.01
    heights = np.arange(zmin+dz/2, zmax-dz/2 + dz/10, dz); massbins = np.zeros_like(heights)
    for i in range(len(zpos)):
        index = int((zpos[i]-zmin)/dz)
        massbins[index] += masses[i]
    # return heights, massbins
    maxid = np.argmax(massbins)
    h1 = np.nan; h2 = np.nan
    for i in range(maxid, len(massbins)):
        if massbins[i] < 0.1*massbins[maxid]:
            h1 = heights[i]
            break
    for i in range(maxid, 0, -1):
        if massbins[i] < 0.1*massbins[maxid]:
            h2 = heights[i]
            break
    time = il.snapshot.loadHeader(path,snap).get('Time')
    print(f'snap {snap} done')
    return time, (h1-h2)/2


def scaleheight_vs_time(path,center,Ncpu=None):
    N=len(glob.glob1(path,"snapshot_*.hdf5"))
    if Ncpu == None:
        Ncpu=multiprocessing.cpu_count()
    p=Pool(Ncpu)
    result=np.array(p.starmap(scaleheight , [(path,i,center) for i in range(N)]))
    p.close()
    p.join()
    return result[:,0],result[:,1]