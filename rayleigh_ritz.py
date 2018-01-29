#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Shun Arahata

import sympy as sym
import scipy.linalg
# from sympy.matrices import Matrix
from sympy import lambdify, symbols
import numpy as np
import matplotlib.pyplot as plt
import itertools

# Constant
l = 100 * (10 ** (-3))  # m
rho = 2.7 * (10 ** (-3)) * ((10 ** 2) ** 3)  # kg/m^3
h = 1 * (10 ** (-3))  # m
E = 70 * (10 ** 9)  # Pa


class beam(object):
    def __init__(self, is_tapering):
        if (is_tapering):
            print("tapering")
            self.EI = (lambda x: E * h ** 3 / 12 * (1 + 2 * x / l) ** 3)
            self.mu = (lambda x: rho * h * (1 + 2 * x / l))
        else:
            print("not tapering")
            self.EI = (lambda x: E * h ** 3 / 12 * (2) ** 3)
            self.mu = (lambda x: rho * h * 2)

    def calc_k(self, phi_i, phi_j):
        x = symbols('x')
        dphi_i = sym.diff(phi_i, x)
        ddphi_i = sym.diff(dphi_i, x)
        dphi_j = sym.diff(phi_j, x)
        ddphi_j = sym.diff(dphi_j, x)
        k_ij = sym.integrate(self.EI(x) * ddphi_i * ddphi_j, (x, 0, l))
        return float(k_ij)

    def make_k_matrix(self, phi):
        """make k matrix.

        :param phi: phi array
        :return:k matrix
        """
        number_of_phi = len(phi)
        k = [[self.calc_k(phi[i], phi[j]) for j in range(number_of_phi)] for i in range(number_of_phi)]
        return k

    def calc_m(self, phi_i, phi_j):
        x = symbols('x')
        m_ij = sym.integrate(self.mu(x) * phi_i * phi_j, (x, 0, l))
        return float(m_ij)

    def make_m_matrix(self, phi):
        """make m matrix.

        :param phi:phi array
        :return:m
        """
        number_of_phi = len(phi)
        m = [[self.calc_m(phi[i], phi[j]) for j in range(number_of_phi)] for i in range(number_of_phi)]
        return m


def make_phi(largest_order):
    """
    calcutate phi.
    :param largest_order:the number of largest order
    :return: phi
    """
    c1 = symbols('c1')
    c2 = symbols('c2')
    c3 = 1
    x = symbols('x')
    phi = c1 * (x / l) ** 4 + c2 * (x / l) ** 5 + c3 * (x / l) ** largest_order
    ddphi = sym.diff(sym.diff(phi, x), x)
    dddphi = sym.diff(ddphi, x)
    eq1 = ddphi.subs([(x, 0)])
    eq2 = dddphi.subs([(x, 0)])
    eq3 = ddphi.subs([(x, l)])
    eq4 = dddphi.subs([(x, l)])
    phi = phi.subs(sym.solve([eq1, eq2, eq3, eq4], [c1, c2]))
    # print(sym.latex(phi))
    return phi


def make_phi_array(largest_order):
    """make array of phi.
    :param largest_order: the number of largest order
    :return:array of phi
    """
    x = symbols('x')
    phi_array = [1, x / l]
    for i in range(6, largest_order + 1):
        phi_array.append(make_phi(i))

    return phi_array


def solve_determinant(phi, beam):
    """solve determinant

    :param phi:phi array
    :return eig_vec: c array
    """
    k = np.array(beam.make_k_matrix(phi))
    m = np.array(beam.make_m_matrix(phi))
    eig_val, eig_vec = scipy.linalg.eig(k, m)  # 一般固化有値問題を解く

    for i in range(len(eig_vec)):  # 正規化
        eig_vec[:, i] = eig_vec[:, i] / np.linalg.norm(eig_vec[:, i])
    eig_vec = eig_vec.T
    selectors = [x > 0 for x in eig_val.real]
    omega = np.sqrt(list(itertools.compress(eig_val, selectors)))
    eig_vec = list(itertools.compress(eig_vec, selectors))
    omega, eig_vec = (list(t) for t in zip(*sorted(zip(omega, eig_vec))))
    return eig_vec


def make_plot(eq_list):
    """make matplotlib figure

    :param eq_list: list of sympy equation
    :return:
    """
    fig, ax = plt.subplots()
    ax.set_xlim([0, l])
    ax.set_ylim([-1, 1])
    x_vals = np.linspace(0, l, 100)
    x = symbols('x')
    for i in range(len(eq_list)):
        lam_w_i = lambdify(x, eq_list[i], modules=['numpy'])
        y_vals = lam_w_i(x_vals)
        max_y = abs(y_vals).max()
        ax.plot(x_vals, y_vals / max_y, label='mode:' + str(i + 1))
    ax.legend()
    plt.show()


def main(n, is_tapering):
    """

    :param n: number of phi
    :param is_tapering: bool tapering or not
    :return:
    """
    largest_order = n + 3
    phi = make_phi_array(largest_order)
    c = solve_determinant(phi, beam(is_tapering))
    w_list = [sum([c[j][i] * phi[i] for i in range(len(phi))]) for j in range(len(c))]
    make_plot(w_list)


if __name__ == '__main__':
    n = int(input("number of omega "))
    print("Is the beam tapering?")
    dic = {'y': True, 'yes': True, 'n': False, 'no': False}
    while True:
        try:
            inp = dic[input('[Y]es/[N]o? >> ').lower()]
            break
        except:
            pass
        print('Error! Input again.')
    main(n, inp)
