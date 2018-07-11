import numpy as np
from numpy.random import normal
from scipy.constants import pi, elementary_charge, electron_mass, epsilon_0, neutron_mass
from scipy.linalg import toeplitz
from scipy.signal import convolve
from scipy.interpolate import interp1d
from functools import reduce
import matplotlib.pyplot as plt

np.random.seed(0)
def main():
    # parameter
    length, dt, dx, electrodeArea = 0.05, 5e-11, 0.05 / 500, 2.0 * pi * np.sqrt(0.05)
    particleCount, particleWeight = int(1e5), 2e9
    voltage = np.array([250.0, 0.0])

    # init grid
    grid, gridMids = np.linspace(0.0, length, int(length / dx + 1)), np.linspace(0.5 * dx, length - 0.5 * dx, int(length / dx))

    # init particles
    species = [(-elementary_charge,electron_mass,np.vstack([normal(length / 2.0, length / 8.0, particleCount), normal(0.0, 1000.0, particleCount)]).T),
               (+elementary_charge,16*neutron_mass,np.vstack([normal(length / 2.0, length / 8.0, particleCount), normal(0.0, 100.0, particleCount)]).T)]

    # init poisson solver
    phi, roh = np.zeros((int(length / dx) + 1, )), np.zeros((int(length / dx) - 1, ))
    poisson = np.linalg.inv(toeplitz([2.0, -1.0] + [0.0] * (phi.shape[0] - 4))) * dx ** 2 / epsilon_0

    plots = [['Electical Field',[lambda: eField(gridMids)],221],
             ['Potential',[lambda: phi],222],
             ['Charge density',[lambda: roh],223],
             ['Ion and electron density',[(lambda x: lambda: np.histogram(x[:,0],grid)[0])(p) for c,m,p in species],224]]

    # pic cycle
    for i in range(10000):
        # calc roh
        roh = particleWeight / (dx * electrodeArea) * reduce(np.add, [c*np.histogram(p[:, 0], grid)[0] for c,m,p in species])
        for j in range(5): roh = convolve(roh, np.ones((20, )) / 20.0, 'same')

        # calc field
        phi[0], phi[1:-1], phi[-1] = voltage[0], np.dot(poisson, 0.5 * (roh[:-1] + roh[1:])) + grid[1:-1] * (voltage[1] - voltage[0]) / length + voltage[0], voltage[1]
        eField = interp1d(gridMids, -(phi[1:] - phi[:-1]) / dx, bounds_error=False, fill_value=0)

        # update speeds & move particles
        for c,m,p in species:
            p[:,1] += eField(p[:,0]) * dt * c / m
            p[:,0] += p[:,1] * dt

        if not i:
            fig = plt.figure()
            for p in plots:
                ax = fig.add_subplot(p[2])
                ax.set_title(p[0])
                p.append([ax.plot(f())[0] for f in p[1]])
            plt.ion(); plt.show()
            plt.savefig(f"{i}.png")

        # print step
        if not i % 10:
            print('step: {}'.format(i))
            for p in plots:
                for l,f in zip(p[3],p[1]): l.set_ydata(f())
                fig.canvas.draw()

if __name__ == '__main__':
    main()
