import numpy as np
import illustris_python as il
import matplotlib.pyplot as plt

def plot(X,Y,n,label=None,color=None,linewidth=None,linestyle=None,alpha=1):
    """Plots the average of Y over n points against the average of X over n points."""
    XX=[]
    YY=[]
    for i in np.arange(0,len(X),1):
        XX.append(np.average(X[i:i+n]))
        YY.append(np.average(Y[i:i+n]))
    plt.plot(XX,YY,label=label,color=color,linewidth=linewidth,linestyle=linestyle,alpha=alpha)
    
def axplot(axis,X,Y,n,label=None,color=None,linewidth=None,linestyle=None,alpha=1):
    """Plots the average of Y over n points against the average of X over n points."""
    XX=[]
    YY=[]
    for i in np.arange(0,len(X),1):
        XX.append(np.average(X[i:i+n]))
        YY.append(np.average(Y[i:i+n]))
    axis.plot(XX,YY,label=label,color=color,linewidth=linewidth,linestyle=linestyle,alpha=alpha)


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