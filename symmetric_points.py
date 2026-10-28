# QPI plots
import os
import numpy as np
import matplotlib.pyplot as plt

def setup_mesh(points, b):
    r_vals = np.arange(points) #+0.01*np.ones(points)
    r1,r2 = np.meshgrid(r_vals,r_vals)
    # b1 = np.array([0,4*np.pi/np.sqrt(3)])
    # b2 = np.array([2*np.pi,2*np.pi/np.sqrt(3)])
    kx = (b[0,0]*r1 + b[1,0]*r2)/points
    ky = (b[0,1]*r1 + b[1,1]*r2)/points
    mesh = np.swapaxes(np.array([kx.flatten(),ky.flatten()]),0,1)
    return np.reshape(mesh,(points,points,2))

def main():
    N = 24
    lattice = np.array([[1.0, 0.0], [-0.5, np.sqrt(3)/2]])
    rlattice = 2*np.pi * np.linalg.inv(lattice).T
    kmesh = setup_mesh(N, rlattice)

    kx = kmesh[..., 0].ravel()
    ky = kmesh[..., 1].ravel()

    m, n = 3, 5
    indices = np.array([
        [n,m],
        [-m,m+n],
        [-m-n,n],
        [-n,-m],
        [m,-n-m],
        [m+n,-n],
        [n,-m-n],
        [-m,-n],
        [-m-n,m],
        [-n,m+n],
        [m,n],
        [m+n,-m]
    ])

    plt.figure(figsize=(6, 6))
    plt.scatter(kx, ky, s=10, c='#f0f0f0', marker='o', edgecolors='none', alpha=0.3)
    for i in range(12):
        p = indices[i,:]
        plt.scatter(kmesh[p[0],p[1],0], kmesh[p[0],p[1],1], s=10, marker='o', label=f'{i}')
    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title(f'k-mesh ({N}x{N})')
    plt.grid(alpha=0.4, linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()



if __name__=='__main__':
    main()