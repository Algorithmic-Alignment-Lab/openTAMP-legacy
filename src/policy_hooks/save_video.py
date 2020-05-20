import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import sys
from matplotlib.animation import FuncAnimation
import imageio


SAVEDIR = '/home/michaelmcdonald/Dropbox/videos/'
def save_video(fname):
    arr = np.load(fname)
    vname = SAVEDIR+fname.split('/')[-1].split('.')[0]+'.gif'
    imageio.mimsave(vname, arr, duration=0.01)
    '''
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    def update(i):
        print('TS', i)
        ax.imshow(arr[i])
    anim = FuncAnimation(fig, update, frames=range(0, len(arr)), interval=100)
    anim.save(vname, dpi=80, writer='imagemagick')
    '''

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if os.path.isdir(sys.argv[1]):
            for f in os.listdir(sys.argv[1]):
                if not f.endswith('npy'): continue
                save_video(sys.argv[1]+'/'+f)
        else:
            save_video(sys.argv[1])
