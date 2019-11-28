# -*- coding: utf-8 -*-
"""
@author: verlo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
import scipy.ndimage.measurements as measurements
from scipy.optimize import curve_fit

class forest():
    "Class defininf the pbject forest"
    
    def __init__(self, p_regrowth=0.05, p_fire=0.001):
        "Initiliase object with empty array, given growing and burning probabilities and initial density of trees"
        
        # The initial fraction of the forest occupied by trees.
        forest_fraction = 0.6
        self.neighbourhood = ((-1,-1), (-1,0), (-1,1), (0,-1), (0, 1), (1,-1), (1,0), (1,1))
        self.EMPTY, self.TREE, self.FIRE = 0, 1, 2
        # Colours for visualization: brown for EMPTY, dark green for TREE and orange
        # for FIRE. Note that for the colormap to work, this list and the bounds list
        # must be one larger than the number of different values in the array.
        colors_list = [(0.2,0,0), (0,0.5,0), (1,0,0), 'r']
        self.cmap = colors.ListedColormap(colors_list)
        bounds = [0,1,2,3]
        self.norm = colors.BoundaryNorm(bounds, self.cmap.N)
        # Probability of new tree growth per empty cell, and of lightning strike.
        self.p, self.f = p_regrowth, p_fire
        # Forest size (number of cells in x and y directions).
        self.nx, self.ny = 100, 100
        # Initialize the forest grid.
        self.X  = np.zeros((self.ny, self.nx))
        self.X[1:self.ny-1, 1:self.nx-1] = np.random.randint(0, 2, size=(self.ny-2, self.nx-2))
        self.X[1:self.ny-1, 1:self.nx-1] = np.random.random(size=(self.ny-2, self.nx-2)) < forest_fraction
        self.number_fires = 0
        self.number_burnt_trees=0
        self.all_clusters=[]
        self.counting_sizes=np.zeros(int(self.nx*self.ny/forest_fraction)+1)
        self.radius=np.zeros(int(self.nx*self.ny/forest_fraction)+1)
        # Displacements from a cell to its eight nearest neighbours
    def cluster_size(self,updated_X):
        "Calculate cluster size, counting them and radius from the centre of mass of each cluster"
        
        Cluster,num=measurements.label(updated_X,structure=[[1,1,1],[1,1,1],[1,1,1]])
        cluster_size=measurements.sum(updated_X,Cluster,index=np.arange(Cluster.max())+1)
        centre_of_mass=measurements.center_of_mass(updated_X,Cluster,index=np.arange(Cluster.max())+1)
        #print(len(cluster_size),num)
        for i in range(1,num):
            #print(i)
            #print(cluster_size[i])
            self.counting_sizes[int(cluster_size[i])]+=1
            if(cluster_size[i] not in self.all_clusters):
                #print(i)
                #print(cluster_size[i])
                self.all_clusters.append(int(cluster_size[i]))
            #centre_of_mass=measurements.center_of_mass(updated_X,Cluster,i)
            #print(centre_of_mass)
            #break
            sum_radii=0
            max_radii=0
            for j in range(1,self.nx-1):
                for k in range(1,self.ny-1):
                    if(Cluster[j][k]==i):
                        sum_radii=((j-centre_of_mass[i][0])**2+(k-centre_of_mass[i][1])**2)**0.5
                        if(sum_radii>max_radii):
                            max_radii=sum_radii
                        #print(sum_radii)
            self.radius[int(cluster_size[i])]+=(max_radii)#update list of radius for given size

    def iterate(self,X):
        """Iterate the forest according to the forest-fire rules."""
    
        # The boundary of the forest is always empty, so only consider cells
        # indexed from 1 to nx-2, 1 to ny-2
        #innit()
        X1 = np.zeros((self.ny, self.nx))
        for ix in range(1,self.nx-1):
            for iy in range(1,self.ny-1):
                if X[iy,ix] == self.EMPTY and np.random.random() <= self.p:
                    X1[iy,ix] = self.TREE
                if X[iy,ix] == self.TREE:
                    X1[iy,ix] = self.TREE
                    for dx,dy in self.neighbourhood:
                        if X[iy+dy,ix+dx] == self.FIRE:
                            X1[iy,ix] = self.FIRE
                            #self.number_burnt_trees+=1
                            break
                    else:
                        if np.random.random() <= self.f:
                            X1[iy,ix] = self.FIRE
                            self.number_fires+=1
                            #self.number_burnt_trees+=1
        return X1
    
    
    # The animation function: called to produce a frame for each generation.
    def animate(self,i):
        "Animate forest trying to show fractal"
        #innit()
      #  X=self.X
        self.im.set_data(self.animate_X)
        self.animate_X = self.iterate(self.animate_X)
    # Bind our grid to the identifier X in the animate function's namespace.
        
    
    def count(self):
        """
        Count number of current trees.
        returns: (no. of unburnt trees, no. of burning trees, no. of blanks)
        """
        n_blnk, n_tr, n_brn = 0, 0, 0
        for row in self.X[1:-1]:
            for state in row[1:-1]:
                if state == 0: n_blnk += 1
                if state == 1: n_tr   += 1
                if state == 2: n_brn  += 1
        
        return n_blnk, n_tr, n_brn

def show_animation():
    current_forest=forest()
    current_forest.animate_X = current_forest.X
    fig = plt.figure(figsize=(25/3, 6.25))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    current_forest.im = ax.imshow(current_forest.X, cmap=current_forest.cmap, norm=current_forest.norm)#, interpolation='nearest')
    # Interval between frames (ms).
    interval = 100
    anim = animation.FuncAnimation(fig, current_forest.animate, interval=interval)
    plt.show()
    return anim


def plot_n_vs_t(steps=1000, p_regrow=0.05, p_burn=0.000001):
    "Function doing the analysis, plotting and fitting of the results"
    
    n_blnk, n_tr, n_brn = [], [], []
    current_forest = forest(p_regrow, p_burn)
    
    for i in range(steps):
        print(steps - i, "remaining")
        blnk, tr, brn = current_forest.count()
        n_blnk.append(blnk)
        n_tr.append(tr)
        n_brn.append(brn)
        current_forest.X = current_forest.iterate(current_forest.X)
        current_forest.cluster_size(current_forest.X)
    n_s=[]
    avg_rad=[]
    current_forest.all_clusters.sort()
    for i in current_forest.all_clusters:
        n_s.append(current_forest.counting_sizes[i])
        avg_rad.append(current_forest.radius[i]/current_forest.counting_sizes[i])
    
    lin = lambda x, m, c: m*x + c
    x_size=np.log(np.array(current_forest.all_clusters))
    y_n=np.log(np.array(n_s))
    y_r=np.log(np.array(avg_rad))
    n_s_fit_param, ns_fit_err = curve_fit(lin, x_size, y_n, [1, 1])
    avg_rad_fit_param, avg_rad_ft_err = curve_fit(lin, x_size, y_r, [-1, 1])
    
    n_s_std_dev = np.sqrt(np.diag(ns_fit_err))
    avg_rad_std_dev = np.sqrt(np.diag(avg_rad_ft_err))
    
    n_s_fit_plot = lin(x_size, *n_s_fit_param) # Y values for plotting the fit
    avg_rad_fit_plot = lin(x_size, *avg_rad_fit_param) # Y values for plotting the fit
    
    plt.plot(x_size, y_n, label="log(N(s))")
    plt.plot(x_size,n_s_fit_plot, label="Fit log(N(s))")
    plt.xlabel('log(s)') 
    plt.ylabel('log(N(s))') 
    plt.legend()
    plt.show()
#    
    plt.plot(x_size, y_r, label="log(R(s))")
    plt.plot(x_size,avg_rad_fit_plot, label="Fit log(R(s))")
    plt.xlabel('log(s)') 
    plt.ylabel('log(R(s))') 
    plt.legend()
    plt.show()
    
    print(n_s_std_dev)
    print(avg_rad_std_dev)
    plt.plot(n_blnk, label="No. Blanks")
    plt.plot(n_tr, label="No. Trees")
    plt.plot(n_brn, label="No. Burning")
    plt.legend()
    plt.show()
    
    x_size=np.log(np.array(current_forest.all_clusters))
    y_n=np.log(np.array(n_s))
    y_r=np.log(np.array(avg_rad))
    fig, ax1 = plt.subplots()
#    x_size=np.array()
    color = 'tab:red'
    ax1.set_xlabel('log(s)')
    ax1.set_ylabel('log(N(s))', color=color)
    ax1.plot(x_size, y_n, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('log(R(s))', color=color)  # we already handled the x-label with ax1
    ax2.plot(x_size, y_r, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    
plot_n_vs_t()

a=show_animation()

        
        
