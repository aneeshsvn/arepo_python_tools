import multiprocessing
from multiprocessing import Pool
import glob
import illustris_python as il
import numpy as np
from global_props import get_particle_data
from com import mediancenter, get_iter_com
import ptorrey_packages.utils.calc_hsml as calc_hsml

def flux_shell(path,snap,R,width,p_type='0',vcut=None,center='bh'):
    '''calculates flux of particle type p_type through a sphere of radius R with given center as sum of mass * radial_vel / shell_width'''
    if center[:2] == 'bh':
        bh = il.snapshot.loadSubset(path,snap,5,['Coordinates','ParticleIDs'])
        if bh['count'] == 1:
            cen = bh['Coordinates'][0]
        else:
            bhids = il.snapshot.loadSubset(path,0,5,'ParticleIDs')
            bhnum = int(center[2:])
            cen = bh['Coordinates'][bh['ParticleIDs']==bhids[bhnum]]
    elif center == 'com':
        # cen = mediancenter(path,snap)
        boxsize = il.snapshot.loadHeader(path,snap).get('BoxSize')
        cen = get_iter_com(path,snap,[boxsize/2,boxsize/2,boxsize/2],100)
    particles = get_particle_data(path,snap,p_type,fields=['Coordinates','Masses','Velocities'])
    time = il.snapshot.loadHeader(path,snap).get('Time')
    if particles['count'] == 0:
        return time, 0
    n=particles['Coordinates']-cen
    r=np.linalg.norm(n,axis=1)
    mask = (r > R - width/2) & (r < R + width/2)
    vel=particles['Velocities'][mask]
    mass=particles['Masses'][mask]
    r=r[mask]
    n=n[mask]
    n=np.array([n[i]/r[i] for i in range(len(n))])
    rvel=np.array([np.dot(vel[i],n[i]) for i in range(len(n))])
    if vcut == None:
        flux = np.sum(mass*rvel)/width
    else:
        mask= (rvel > vcut)
        flux = np.sum(mass[mask]*rvel[mask])/width
    return time, flux, len(rvel)

def flux_vs_time_shell(path,R,width,p_type='0',Ncores=None,vcut=None,center='bh0',normalize=False):
    N=len(glob.glob1(path,"snapshot*.hdf5"))
    if Ncores == None:
        Ncores = multiprocessing.cpu_count()
    p=Pool(Ncores)
    args = [(path,snap,R,width,p_type,vcut,center) for snap in range(N)]
    result = np.array(p.starmap(flux_shell, args))
    p.close()
    p.join()
    fac = 1
    if normalize == True:
        particles = get_particle_data(path,0,p_type,fields=['Coordinates','Masses'])
        if center[:2] == 'bh':
            cen = il.snapshot.loadSubset(path,0,5,'Coordinates')[int(center[2:])]
        elif center == 'median':
            cen = mediancenter(path,0)
        r = np.linalg.norm(particles['Coordinates']-cen,axis=1)
        fac = np.sum(particles['Masses'][r<=R])
    return result[:,0], result[:,1]/fac, result[:,2]

def flux_surf_int(path,snap,R,Npoints,ngb=12):
    '''calculates flux of particle type p_type through a sphere of radius R with given center using surface integral method'''
    time=il.snapshot.loadHeader(path,snap).get('Time')
    gas=il.snapshot.loadSubset(path,snap,0,fields=['Coordinates','Masses','Velocities'])
    bhpos=il.snapshot.loadSubset(path,snap,5,'Coordinates')[0]
    x=gas['Coordinates']-bhpos
    m=gas['Masses']
    v=gas['Velocities']
    r=np.linalg.norm(x,axis=1)
    n=np.array([x[i]/r[i] for i in range(len(r))])
    rvel=np.array([np.dot(v[i],n[i]) for i in range(len(n))])
    quant=m*rvel
    crd=[[0,0,R],[0,0,-R]]
    theta = np.linspace(0,np.pi,Npoints)
    dl = R*(theta[1]-theta[0])
    dA = [np.pi*dl**2,np.pi*dl**2]
    z = np.cos(theta[1:-1])*R
    z1 = (z[:-1]+z[1:])/2
    for z11 in z1:
        dphi = dl/np.sqrt(R**2-z11**2)
        dphi = 2*np.pi / int(2*np.pi/dphi)
        phi = np.arange(0,2*np.pi,dphi)
        for phi1 in phi:
            crd.append([R*np.cos(phi1),R*np.sin(phi1),z11])
            dA.append(dl*R*dphi)
    crd=np.array(crd)
    dA = np.array(dA)
    radmomdens=calc_hsml.get_gas_density_around_stars( x[:,0], x[:,1], x[:,2], quant, crd[:,0], crd[:,1], crd[:,2], DesNgb=ngb)
    outflowrate=np.sum(radmomdens*dA)
    return time,outflowrate

def flux_vs_time_surf_int(path,R,Npoints,Ncores=None,normalize=False):
    N=len(glob.glob1(path,"snapshot*.hdf5"))
    if Ncores == None:
        Ncores = multiprocessing.cpu_count()
    p=Pool(Ncores)
    args = [(path,snap,R,Npoints) for snap in range(N)]
    result = np.array(p.starmap(flux_surf_int, args))
    p.close()
    p.join()
    fac = 1
    if normalize == True:
        gas = il.snapshot.loadSubset(path,0,0,['Coordinates','Masses'])
        cen = mediancenter(path,0)
        r = np.linalg.norm(gas['Coordinates']-cen,axis=1)
        fac = np.sum(gas['Masses'][r<=R])
    return result[:,0], result[:,1]/fac

