import sys
import site
site.addsitedir('/home/aneeshs/Packages')
sys.path.append('/home/aneeshs/Packages/illustris_python')
sys.path.append('/home/aneeshs/Packages/arepo_python_tools')
from multiprocessing import Pool
import glob
import illustris_python as il
import arepo_tools as ap
import simulation_paths as paths
import matplotlib
matplotlib.use('Agg')
from multiprocessing import Pool
# import timeit
import numpy as np
import os
import re

def makeplot(path,snap):
        savename=path+'/uva_movie2_snap{:03d}.png'.format(snap) # path to save the plot
        # make plot from the snapshot
        ap.galaxy2Dplots(path,snap,'0','Density','xy',60,180,180,3000,fill=1,showBH=False,colorbar=False,scalelen=35,save_name=savename,dpi=600)

def makemovie(path,restart=False):
    print('Plotting '+path)
    N=len(glob.glob1(path,"snapshot*.hdf5"))
    if restart == True:
        #find missing plots in the directory and make only those plots
        files = os.listdir(path)
        snaps = set(range(0,N))
        # snaps1 = set(range(N))
        xz_pattern = r'uva_movie2_snap(\d+)\.png'
        for file in files:
            xz_match = re.search(xz_pattern, file)
            if xz_match:
                index = int(xz_match.group(1))
                snaps -= {index}
    else:
         snaps = range(0,N)
    print('snaps = ',snaps)

    print(f'N = {N}')
    ntask = os.cpu_count()
    print(f'ntask = {ntask}')
    p=Pool(ntask)
    args = [(path,snap) for snap in snaps]
    p.starmap(makeplot,args)
    p.close()
    p.join()
    print('All snapshots plotted.')
    os.system('ffmpeg -y -r 16 -start_number 0 -i {}/uva_movie2_snap%03d.png -c:v libx264 -pix_fmt yuv420p -vf scale=-2:720 {}/../uva_movie2.mp4'.format(path,path))
    os.system('rm {}/uva_movie2_snap*.png'.format(path))
    


makemovie(paths.mergermw_M_Nofb_alpha5,restart=False)

