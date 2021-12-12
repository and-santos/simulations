# -*- coding: utf-8 -*-


import numpy as np
from scipy.special import eval_genlaguerre as lg
import matplotlib.pyplot as plt


class Simulation:

    def __init__(self, l, p=0, A=1):
        self.lamb = 1064e-9  # [m]
        self.w0 = 200e-6  # [m]
        self.k0 = 2 * np.pi / self.lamb
        self.zR = self.k0 * self.w0 ** 2 / 2  # [ 0.12 m]
        self.L = 200  # (número de pontos no plano xy )
        self.N = 300  # (número de pontos ao longo do eixo z)
        self.p = p
        self.l = l
        self.A = A
        self.c = 3 * 10 ** 8  # [m/s]
        self.omega = self.c * self.k0
        self.chi3 = 1e-22
        self.R = 2 / (np.pi * self.w0 ** 2)
        self.g = (3 * self.omega ** 2 * self.chi3 *
                  self.R) / 2 * self.k0 * self.c
        self.generate_grid()
        self.superposition_plot(3 * 12e-2, 1e57)

    def generate_grid(self):
        self.x = np.linspace(-10 * self.w0, 10 * self.w0, self.L)  # [mm]
        self.y = np.linspace(-10 * self.w0, 10 * self.w0, self.L)  # [mm]
        self.X, self.Y = np.meshgrid(self.x, self.y)  # [mm]

        self.r = np.sqrt(self.X ** 2 + self.Y ** 2)  # [mm]
        self.phi = np.arctan2(self.Y, self.X)  # [adimensional]

    def superposition_plot(self, zobs, saturation=None):
        waist = self.w0 * np.sqrt(1 + (zobs / self.zR) ** 2)
        self.r = self.r / waist

        T1 = self.A * (np.sqrt(2) * self.r) ** abs(self.l)
        T2 = np.exp(-(1 + 1j * zobs / self.zR) * self.r ** 2)
        T3 = np.exp(1j * (2 * self.p + abs(self.l) + 1)
                    * np.arctan(zobs / self.zR))
        T4 = np.exp(1j * self.l * self.phi)
        self.modolg = T1 * T2 * T3 * T4 * \
            lg(self.p, abs(self.l), 2 * self.r ** 2)

        waist_renorm = self.w0 * np.sqrt(1 + (zobs) / self.zR ** 2)
        self.r_norm = self.r / waist_renorm
        T1n = self.A * (np.sqrt(2) * self.r) ** abs(self.l)
        T2n = np.exp(-(1 + 1j * zobs / self.zR) * self.r ** 2)
        T3n = np.exp(1j * (2 * self.p + abs(self.l) + 1)
                     * np.arctan(zobs / self.zR))
        T4n = np.exp(1j * self.l * self.phi)
        cte = (self.g * self.A ** 3) / 3 ** ((3 * abs(self.l) + 1) / 2)
        vortice = self.modolg
        for p in range(1, abs(self.l) + 1):
            vortice += 1j * zobs * cte * (T1n * T2n * T3n * T4n *
                                          lg(p, abs(self.l), 2 * self.r ** 2))

        plt.imshow(abs(vortice) ** 2, cmap='gray', vmax=saturation)
        plt.axis('off')
        plt.show()


def main():
    # ( l, A):
    # simulator = Simulation(0)
    # simulator = Simulation(1)
    # simulator = Simulation(2)
    simulator = Simulation(3)


if __name__ == "__main__":
    main()
