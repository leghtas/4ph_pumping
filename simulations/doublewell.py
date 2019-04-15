from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


plt.close('all')

#ax3 = fig.gca(projection='3d')
# Make data.
alphamax=np.sqrt(10)
X = np.arange(-3, 3, 0.05)
Y = np.arange(-3, 3, 0.05)
X, Y = np.meshgrid(X, Y)
#Z = -np.exp(-((X-2.5)**2 + Y**2))*np.exp(-((X+2.5)**2 + Y**2))
alpha = X + 1j*Y
#Z = np.abs((alpha**2-alpha0**2)*alpha)
k2 = 1
alpha0 = np.sqrt(8)
e1 = 3.079 #1ph drive
k1 = 0 #1ph dissipation

def getZ(alpha0):
    e2 = k2/2*alpha0**2
    Z = k2/4*(X**4+Y**4)+k2/2*(X**2*Y**2)+e2*(-X**2+Y**2)-e1*X+k1/4*(X**2+Y**2)
    Z -= np.min(Z)
    Z[Z>2*zmax]=np.nan
    return Z

def getCond(alpha0):
    e2 = k2/2*alpha0**2
    _X = X[0]
    sol = np.sqrt((2*e2-k1/2)/3/k2)
    def f(x):
        return k2*x**3-2*e2*x-e1+k1/2*x
    delta = f(-sol)-f(sol)
    delta = sol*(4/3*e2-1/3*k1)
    return k2*_X**3-2*e2*_X-e1+k1/2*_X, delta

zmax = 15

my_cmap = cm.hot

# Plot the surface.
def plotX(alpha0):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(40, 90)
    surf = ax.plot_surface(X, Y, getZ(alpha0)+1.2, cmap=my_cmap,
                           linewidth=1, antialiased=True, alpha=0.4, vmin=0, vmax=zmax)
    levels=list(np.linspace(0,10,31))
    cset = ax.contourf(X, Y, getZ(alpha0), levels, cmap=my_cmap, alpha=1, zdir='z', offset=0, vmin=0, vmax=zmax)
    ax.set_zlim(0, zmax)
    ax.set_xticklabels(())
    #ax.set_yticklabels(())
    ax.set_zticklabels(())
    plt.savefig('doublewell2 %s .png' % str(round(alpha0**2)))

#plotX(np.sqrt(2))
#plotX(np.sqrt(4))
#plotX(np.sqrt(6))


#
#Xa = np.arange(-5, 5, 1)
#Ya = np.arange(-5, 5, 1)
#Xa, Ya = np.meshgrid(Xa, Ya)
#Za = -np.exp(-((Xa-2.5)**2 + Ya**2))-np.exp(-((Xa+2.5)**2 + Ya**2))
#Xada = Xa+0.01
#Yada = Ya+0.01
#Zada = -np.exp(-((Xada-2.5)**2 + Yada**2))-np.exp(-((Xada+2.5)**2 + Yada**2))
#ax.quiver(Xa, Ya, Za, Xada-Xa, Yada-Ya, Zada-Za, length=10, normalize=False,
#          arrow_length_ratio=1)




alpha0 = np.sqrt(4)
plotX(alpha0)
fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].pcolor(X, Y, getZ(alpha0)+1, cmap=my_cmap, vmin=0, vmax=zmax)
ax[0].set_aspect('equal')
ax[0].set_ylim((-1.5,1.5))
plot, delta = getCond(alpha0)
ax[1].plot(X[0], plot)
print(delta)
xlim =  ax[1].get_xlim()
ax[1].hlines(0,xlim[0], xlim[1], linestyle='dashed')
#ax.set_axis_off()
plt.savefig('doublewell_background.png')
# Customize the z axis.
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


plt.show()
