import numpy as np
from math import cos
from math import sin
from math import pi

def T01(th_i = 0.):
    al_i = pi/2
    a_i  = 0.
    d_i = 16.25
    th_i = th_i
    T01 = np.array([[cos(th_i), -(cos(al_i)*sin(th_i)), sin(al_i)*sin(th_i), a_i*cos(th_i)],
                    [sin(th_i), cos(al_i)*cos(th_i), -(sin(al_i)*cos(th_i)), a_i*sin(th_i)],
                    [0, sin(al_i), cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T01

def T12(th_i = 0.):
    al_i = 0.
    a_i  = -42.50
    d_i = 0
    th_i = th_i
    T12 = np.array([[cos(th_i), -(cos(al_i)*sin(th_i)), sin(al_i)*sin(th_i), a_i*cos(th_i)],
                    [sin(th_i), cos(al_i)*cos(th_i), -(sin(al_i)*cos(th_i)), a_i*sin(th_i)],
                    [0, sin(al_i), cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T12

def T23(th_i = 0.):
    al_i = 0.
    a_i  = -39.22
    d_i = 0.
    th_i = th_i
    T23 = np.array([[cos(th_i), -(cos(al_i)*sin(th_i)), sin(al_i)*sin(th_i), a_i*cos(th_i)],
                    [sin(th_i), cos(al_i)*cos(th_i), -(sin(al_i)*cos(th_i)), a_i*sin(th_i)],
                    [0, sin(al_i), cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T23

def T34(th_i = 0.):
    al_i = pi/2
    a_i  = 0.
    d_i = 13.33
    th_i = th_i
    T34 = np.array([[cos(th_i), -(cos(al_i)*sin(th_i)), sin(al_i)*sin(th_i), a_i*cos(th_i)],
                    [sin(th_i), cos(al_i)*cos(th_i), -(sin(al_i)*cos(th_i)), a_i*sin(th_i)],
                    [0, sin(al_i), cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T34

def T45(th_i = 0.):
    al_i = -(pi/2)
    a_i  = 0.
    d_i = 9.97
    th_i = th_i
    T45 = np.array([[cos(th_i), -(cos(al_i)*sin(th_i)), sin(al_i)*sin(th_i), a_i*cos(th_i)],
                    [sin(th_i), cos(al_i)*cos(th_i), -(sin(al_i)*cos(th_i)), a_i*sin(th_i)],
                    [0, sin(al_i), cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T45

def T56(th_i = pi): #0
    al_i = 0.
    a_i  = 0.
    d_i = 9.96
    th_i = th_i
    T56 = np.array([[cos(th_i), -(cos(al_i)*sin(th_i)), sin(al_i)*sin(th_i), a_i*cos(th_i)],
                    [sin(th_i), cos(al_i)*cos(th_i), -(sin(al_i)*cos(th_i)), a_i*sin(th_i)],
                    [0, sin(al_i), cos(al_i), d_i],
                    [0, 0, 0, 1]])
    return T56


# def T(DH_parameter = [0., 0., 0., 0.]):
#     al_i = DH_parameter[0]
#     a_i  = DH_parameter[1]
#     d_i = DH_parameter[2]
#     th_i = DH_parameter[3]
#     T = np.array([[cos(th_i), -(cos(al_i)*sin(th_i)), sin(al_i)*sin(th_i), a_i*cos(th_i)],
#                     [sin(th_i), cos(al_i)*cos(th_i), -(sin(al_i)*cos(th_i)), a_i*sin(th_i)],
#                     [0, sin(al_i), cos(al_i), d_i],
#                     [0, 0, 0, 1]])
#     return T
#
# def T01(th_i = 0.):
#     parameter = [pi/2, 0., 16.25, th_i]
#
#     return T(parameter)
#
#
# def T12(th_i=0.):
#     parameter = [0., -42.50, 0., th_i]
#
#     return T(parameter)
#
#
# def T23(th_i=0.):
#     parameter = [0., -39.22, 0, th_i]
#
#     return T(parameter)
#
#
# def T34(th_i=0.):
#     parameter = [pi / 2, 0., 13.33, th_i]
#
#     return T(parameter)
#
#
# def T45(th_i=0.):
#     parameter = [- (pi / 2), 0., 9.97, th_i]
#
#     return T(parameter)
#
# def T56(th_i=0.):
#     parameter = [0., 0., 9.96, th_i]
#
#     return T(parameter)