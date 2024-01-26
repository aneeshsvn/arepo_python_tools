# Description: This file contains functions to read blackhole properties from the simulation data
import numpy as np
import illustris_python as il
import multiprocessing
from multiprocessing import Pool
import operator
import glob


def bhproperty(path,bhnum,bhproperty,numpoints=None):
    """Extracts specified property of BHs as a function of time from blackhole_details.txt files.
    Parameters:
    - path (str): The path to the simulation directory.
    - bhnum (int): The number of the BH to extract properties for (0 or 1).
    - bhproperty (str): The property to extract (rho, cs, mdot, mass, or windenergy).   
    - numpoints (int, optional): The number of points to average over. Default is None.
    Returns:
    - time (array-like): All timesteps
    - bhprop (array-like): The specified BH property at each timestep."""

    bh_details = []
    if bhproperty=='rho':
        index=4
    elif bhproperty=='cs':
        index=5
    elif bhproperty=='mdot':
        index=3
    elif bhproperty=='mass':
        index=2
    elif bhproperty=='windenergy':
        index=6
    else:
        print('bhproperty can only be {rho, cs, mdot, mass}')
        return
        
    for filename in glob.glob(path+'/blackhole_details/blackhole_details_*.txt'):
        tmp=np.genfromtxt(filename,dtype=None,encoding=None)
        if len(bh_details)==0:
            bh_details=tmp
        else:
            bh_details=np.concatenate((bh_details,tmp),axis=0)
    bh_details=sorted(bh_details, key=operator.itemgetter(0,1))
    length=len(bh_details)
    idx=length-1
    for i in np.arange(0,length-2):
        if bh_details[i+1][0]!=bh_details[i][0]:
            idx=i+1
            break
    if bhnum==0:
        if numpoints==None:
            time=np.array([x[1] for x in bh_details[0:idx]])
            bhprop=np.array([x[index] for x in bh_details[0:idx]])
        else:
            bins=int(idx/numpoints)
            time=np.array([np.average([x[1] for x in bh_details[bins*i:bins*(i+1)]]) for i in np.arange(0,numpoints)])
            bhprop=np.array([np.average([x[index] for x in bh_details[bins*i:bins*(i+1)]]) for i in np.arange(0,numpoints)])
        return time,bhprop
    elif bhnum==1:
        if numpoints==None:
            time=np.array([x[1] for x in bh_details[idx:length]])
            bhprop=np.array([x[index] for x in bh_details[idx:length]])
        else:
            bins=int((length-idx-1)/numpoints)
            time=np.array([np.average([x[1] for x in bh_details[(idx+bins*i):(idx+bins*(i+1))]]) for i in np.arange(0,numpoints)])
            bhprop=np.array([np.average([x[index] for x in bh_details[(idx+bins*i):(idx+bins*(i+1))]]) for i in np.arange(0,numpoints)])
        return time,bhprop
    else:
        print('bhnum can only take 0 or 1')


def BHFWdetails(path):
    """Extracts BH fast wind injection details from blackhole_FW*.txt files."""
    bhdetails = np.array([])
    for filename in glob.glob(path+'/blackhole_details/blackhole_FW*.txt'):
        tmp=np.loadtxt(filename,dtype={'names': ('bh_pid', 'time', 'windenergy', 'totenergy', 'prob', 'mass', 'density', 'nx', 'ny', 'nz', 'r', 'h_i'),'formats': ('U25', 'f8', 'f8','f8','f8','f8','f8','f8','f8','f8','f8','f8')})
        if tmp.size !=0:
            if bhdetails.size == 0:
                bhdetails=tmp
            else:
                bhdetails=np.hstack((bhdetails,tmp))
    bhdetails=np.sort(bhdetails, order=['time'])
    bhids=il.snapshot.loadSubset(path,0,5,fields=['ParticleIDs'])
    return [bhdetails[bhdetails['bh_pid']==f'BH={bhids[i]}'] for i in range(len(bhids))]

def bhprop_snap_temp(path, snap, prop):
        time=il.snapshot.loadHeader(path,snap).get('Time')
        bhp=il.snapshot.loadSubset(path,snap,5,fields=[prop,'ParticleIDs'])
        op=[time]
        for i in range(len(bhp['ParticleIDs'])):
            op.extend([bhp['ParticleIDs'][i],bhp[prop][i][0],bhp[prop][i][1],bhp[prop][i][2]])
        return op

def bhprop_snap(path,prop='Coordinates',Ncpus=None):
    """Extracts specified property of all BHs as a function of time from snapshots.
    output is a list of dictionaries where each dictionary is for each BH.
    prop has to be a vector like velocity, position, acceleration, etc."""

    N=len(glob.glob1(path,"snapshot*.hdf5"))
    bhid0=il.snapshot.loadSubset(path,0,5,'ParticleIDs')
    bhlist={bhid0[i]:i for i in range(len(bhid0))}
    output = [{'time':np.array([]),'x':np.array([]),'y':np.array([]),'z':np.array([])} for _ in bhid0]    
    if Ncpus == None:
        Ncpus = multiprocessing.cpu_count()
    args = [(path,snap,prop) for snap in range(N)]
    p=Pool(Ncpus)
    result=p.starmap(bhprop_snap_temp , args)
    for arr in result:
        nos = int((len(arr)-1)/4+0.1)
        for i in range(nos):
            output[bhlist[arr[4*i+1]]]['time']=np.append(output[bhlist[arr[4*i+1]]]['time'],arr[0])
            output[bhlist[arr[4*i+1]]]['x']=np.append(output[bhlist[arr[4*i+1]]]['x'],arr[4*i+2])
            output[bhlist[arr[4*i+1]]]['y']=np.append(output[bhlist[arr[4*i+1]]]['y'],arr[4*i+3])
            output[bhlist[arr[4*i+1]]]['z']=np.append(output[bhlist[arr[4*i+1]]]['z'],arr[4*i+4])
    return output
