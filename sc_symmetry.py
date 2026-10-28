import numpy as np #ignore
import matplotlib.pyplot as plt #ignore
from tqdm import tqdm #ignore
from numpy import fft #ignore

b1 = np.array([4*np.pi/3, 0])
b2 = np.array([2*np.pi/3, 2*np.pi/np.sqrt(3)])

N = 250
r = np.linspace(0, 1, N)
r1, r2 = np.meshgrid(r, r)
kx = r1 * b1[0] + r2 * b2[0]
ky = r1 * b1[1] + r2 * b2[1]

qx = np.copy(kx)
qy = np.copy(ky)

a1 = np.array([1, 0])
a2 = np.array([-1/2, np.sqrt(3)/2])
a3 = np.array([-1/2, -np.sqrt(3)/2])

rx = r1 * a1[0] + r2 * a2[0]
ry = r1 * a1[1] + r2 * a2[1]


dx2y2 = (
    (2/np.sqrt(3))*np.cos(a1[0]*kx + a1[1]*ky) -
    (1/np.sqrt(3))*np.cos(a2[0]*kx + a2[1]*ky) -
    (1/np.sqrt(3))*np.cos(a3[0]*kx + a3[1]*ky)
)
dxy = np.cos(a2[0]*kx + a2[1]*ky) - np.cos(a3[0]*kx + a3[1]*ky)



Mq = np.zeros((N,N), dtype=float)

for ii, jj in tqdm(np.ndindex(N,N),total=N**2):
    dx2y2_q = (
        (2/np.sqrt(3))*np.cos(a1[0]*(kx - qx[ii,jj]) + a1[1]*(ky-qy[ii,jj])) -
        (1/np.sqrt(3))*np.cos(a2[0]*(kx - qx[ii,jj]) + a2[1]*(ky-qy[ii,jj])) -
        (1/np.sqrt(3))*np.cos(a3[0]*(kx - qx[ii,jj]) + a3[1]*(ky-qy[ii,jj]))
    )
    dxy_q = (
        np.cos(a2[0]*(kx - qx[ii,jj]) + a2[1]*(ky-qy[ii,jj])) -
        np.cos(a3[0]*(kx - qx[ii,jj]) + a3[1]*(ky-qy[ii,jj]))
    )

    Mq[ii,jj] = np.sum(dx2y2 * dxy_q - dxy * dx2y2_q)/N**2


plt.figure()
im = plt.imshow(Mq)
plt.colorbar(im)
plt.show()
plt.close()

M = fft.rfft2(Mq)

plt.figure()
im = plt.imshow(np.real(M))
plt.colorbar(im)
plt.show()
plt.close()