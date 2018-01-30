#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:Shun Arahata
""" Numerical Analysis of free-free beam.

This module numerically analyze free-free beam vibration
using Rayleigh-Ritz method.

"""
import sympy as sym
import scipy.linalg
from sympy import lambdify, symbols
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy as sp

# Constant
l = 100 * (10 ** (-3))  # m
rho = 2.7 * (10 ** (-3)) * ((10 ** 2) ** 3)  # kg/m^3
h = 1 * (10 ** (-3))  # m
E = 70 * (10 ** 9)  # Pa
non_dimensionize = 1 / l ** 2 * np.sqrt(E * h ** 3 / 12 * (2) ** 3 / (rho * h * 2))


class Beam(object):
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
        function = self.EI(x) * ddphi_i * ddphi_j
        f = lambdify(x, function, modules=['numpy'])
        k_ij = quad(f, 0, l)[0]
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
        function = self.mu(x) * phi_i * phi_j
        f= lambdify(x, function, modules=['numpy'])
        m_ij = quad(f, 0, l)[0]
        return float(m_ij)

    def make_m_matrix(self, phi):
        """make m matrix.

        :param phi:phi array
        :return:m
        """
        number_of_phi = len(phi)
        m = [[self.calc_m(phi[i], phi[j]) for j in range(number_of_phi)] for i in range(number_of_phi)]
        return m


def make_phi_x(largest_order):
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
    return phi


def make_phi_hyper(n):
    """make phi n th expression
    n must be n<=4

    :param n:
    :return: phi
    """
    x = symbols('x')
    kl = make_kl(n+1)
    k = kl / l
    phi = sym.sinh(k * x) + sym.sin(k * x) + \
          (sp.sin(kl) - sp.sinh(kl)) / (sp.cosh(kl) - sp.cos(kl)) * (sym.cosh(k * x) + sym.cos(k * x))
    return phi


def make_phi_array(n, is_x):
    """ make array of phi

    :param n: the number of phi
    :param is_x: phi is expression of polynomial x
    :return:array of phi
    """
    phi_array = []
    if (is_x):
        x = symbols('x')
        phi_array.append(1)
        phi_array.append(x / l)
        largest_order = n + 3
        for i in range(6, largest_order + 1):
            phi_array.append(make_phi_x(i))

    else:
        for i in range(n):
            phi_array.append(make_phi_hyper(i))
    return phi_array


def solve_determinant(phi, Beam):
    """solve determinant

    :param phi:phi array
    :return eig_vec: c array
    """
    k = np.array(Beam.make_k_matrix(phi))
    m = np.array(Beam.make_m_matrix(phi))
    eig_val, eig_vec = scipy.linalg.eig(k, m)  # 一般固化有値問題を解く

    for i in range(len(eig_vec)):  # 正規化
        eig_vec[:, i] = eig_vec[:, i] / np.linalg.norm(eig_vec[:, i])
    eig_vec = eig_vec.T
    selectors = [x > 0 for x in eig_val.real]
    omega = np.sqrt(list(itertools.compress(eig_val, selectors)))
    eig_vec = list(itertools.compress(eig_vec, selectors))
    omega, eig_vec = (list(t) for t in zip(*sorted(zip(omega, eig_vec))))
    print("omega", omega / non_dimensionize)
    return eig_vec


def make_plot(eq_list, title):
    """make matplotlib figure

    :param eq_list: list of sympy equation
    :param title: title of fig
    :return:
    """
    fig, ax = plt.subplots()
    ax.set_xlim([0, l])
    ax.set_ylim([-2, 2])
    x_vals = np.linspace(0, l, 100)
    x = symbols('x')
    for i in range(3):
        lam_w_i = lambdify(x, eq_list[i], modules=['numpy'])
        y_vals = lam_w_i(x_vals)
        min_y = - y_vals[0]
        ax.plot(x_vals, y_vals / min_y, label='mode:' + str(i + 1))
    ax.legend()
    name = "fig/" + title + ".pgf"
    # plt.show()
    plt.savefig(name)


def main(n, is_tapering, is_x):
    """

    :param n: number of phi
    :param is_tapering: bool tapering or not
    :param is_x: phi is expression of polynomial x
    :return:
    """
    phi = make_phi_array(n, is_x)
    c = solve_determinant(phi, Beam(is_tapering))
    w_list = [sum([c[j][i] * phi[i] for i in range(len(phi))]) for j in range(len(c))]
    title = str(n) + str(inp)
    make_plot(w_list, title)


def make_kl(n):
    """calculate coshx*cosx=1 (radian)
    using　periodicity　of cosx

    :param n:nth answer
    :return:
    """
    product = [ abs(sp.cosh(x)*sp.cos(x)-1) for x in np.arange(n*np.pi, (n+1/2)*np.pi,0.001)]
    index = product.index(min(product))
    kl = n*np.pi+0.001*index
    return kl


if __name__ == '__main__':
    print("Is the Beam tapering?")
    dic = {'y': True, 'yes': True, 'n': False, 'no': False}
    while True:
        try:
            inp = dic[input('[Y]es/[N]o? >> ').lower()]
            break
        except:
            pass
        print('Error! Input again.')
    print("Phi is polynomial expression　of x ?")
    while True:
        try:
            exp = dic[input('[Y]es/[N]o? >> ').lower()]
            break
        except:
            pass
        print('Error! Input again.')

    if (exp):
        n = int(input("number of phi "))
        main(n, inp, 1)
    else:
        n = int(input("number of phi "))
        main(n, inp, 0)
