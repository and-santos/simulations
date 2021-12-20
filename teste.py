self.generate_grid()
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