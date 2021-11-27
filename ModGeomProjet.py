#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" Code by Manon Vialle, Rahab Lacroix Thomas and Bréhault Gabriel "
" Update on : 22/11/2021                                         "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

fig2 = plt.figure()
ax = fig2.add_subplot(111)
minmax = 10
ax.set_xlim((-minmax,minmax))
ax.set_ylim((-minmax,minmax))
plt.title("Left click to add/remove point, right click to update bezier curve")
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
points, = plt.plot([], [], 'bx')
poly, = plt.plot([], 'r')
curve, = plt.plot([], 'g')

#-----------
#factorial n
def fact(n):
    res = 1
    for i in range(n):
        res *= (i+1)
    return res

#----------------------
# Binomial coefficient
def nchoosek(n,k):
    return fact(n)/(fact(k)*fact(n-k))

#---------------------
# Bernstein Polynomials
# N is the degree
# t = np.linspace(0,1,500)
def Bernstein(N,t):
    BNt = np.zeros((N+1, t.size))
    for i in range(N+1):
         BNt[i, :] = (nchoosek(N, i)*(t**i)*(1-t)**(N-i) )  
    return BNt

#----------------------
# plot of the Bernstein polynomials
def plotBernPoly():
    N=5
    x = np.linspace(0,1,500)
    Bern = Bernstein(N,x)
    for k in range(N+1):
        plt.plot(x,Bern[k, :])

#--------------------------
# plot of the Bezier curve
def PlotBezierCurve(Polygon):
    N = len(Polygon[0, :])-1
    t = np.linspace(0,1,500)
    Bern = Bernstein(N, t)
    Bezier = Polygon @ Bern
    curve.set_xdata(Bezier[0,:])
    curve.set_ydata(Bezier[1,:])
    plt.draw()
    return


def AcquisitionNvxPoints(minmax,color1,color2):
    x = points.get_xdata().copy()
    y = points.get_ydata().copy()
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        if coord != []:
            plt.draw()
            xx = coord[0][0]
            yy = coord[0][1]
            points.set_xdata(np.append(points.get_xdata(), xx))
            points.set_ydata(np.append(points.get_ydata(), yy))
            x = np.append(x, xx)
            y = np.append(y, yy)
            poly.set_xdata(points.get_xdata())
            poly.set_ydata(points.get_ydata())
    #Polygon creation
    Polygon = np.zeros((2,len(x)))
    Polygon[0,:] = x
    Polygon[1,:] = y
    return Polygon


def AcquisitionRMVPoints(minmax,color1,color2):
    x = points.get_xdata().copy()
    y = points.get_ydata().copy()
    coord = 0
    while coord != []:
        coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        if coord != []:
            plt.draw()
            xx = coord[0][0]
            yy = coord[0][1]
            dist = [(xx-x[i])**2+(yy-y[i])**2 for i in range(x.size)]
            index_min = min(range(len(dist)), key=dist.__getitem__)
            points.set_xdata(np.delete(points.get_xdata(), index_min))
            points.set_ydata(np.delete(points.get_ydata(), index_min))
            x = np.delete(x, index_min)
            y = np.delete(y, index_min)
            poly.set_xdata(points.get_xdata())
            poly.set_ydata(points.get_ydata())
    #Polygon creation
    Polygon = np.zeros((2,len(x)))
    Polygon[0,:] = x
    Polygon[1,:] = y
    return Polygon





#notre travail

def WhatIsC():
    print("Merci de retourner une valeur appartenant à [0;1]")
    return float(input("Vous choisissez c = "))


def WhatAreMoMn(Polygon):
    m0 = np.zeros((2, 1))
    mn = np.zeros((2, 1))
    choice = input('Voulez-vous choisir les M0 et Mn ? (Non/oui) : ')
    if choice.lower() == 'oui':
        #ecire des trucs intelligen
        return m0, mn
    #eciree des quetru intilleierz aehgj
    return m0, mn

def PlotHermiteCurve(Polygon):
    c = WhatIsC()
    # m0, mn = WhatAreMoMn(Polygon, c)
    m0 = [3, 3]
    mn = [7, 7]
    N = len(Polygon[0, :])-1
    t = np.linspace(0,1,10)
    # Hermi = Hermite(N, t, m0, mn, c, Polygon)
    # Spline = Polygon @ Hermi
    Spline = Hermite(N, t, m0, mn, c, Polygon)
    curve.set_xdata(Spline[0,:])
    curve.set_ydata(Spline[1,:])
    plt.draw()
    return


def Mk_List(N, m0, mn, c, Polygon):
    Liste = np.zeros((2, N+1))
    Liste[0, 0] = m0[0]
    Liste[1, 0] = m0[1]
    Liste[0, N] = mn[0]
    Liste[1, N] = mn[1]

    for i in range(1, N):
        Liste[0, i] = (1-c)*(Polygon[0, i+1] - Polygon[0, i-1])/2
        Liste[0, i] = (1-c)*(Polygon[1, i+1] - Polygon[1, i-1])/2
    return Liste


def Hermite(N, T, m0, mn, c, Polygon):
    
    mk_list = Mk_List(N, m0, mn, c, Polygon)
    Hrmt = np.zeros((2, (N+1)*T.size))

    for i in range(N):
        for t in range (T.size + 1):
            
            Hrmt[0, i*T.size + t] = (Polygon[0, i])*((1 - t)**2)*(1 + 2*t) + (Polygon[0, i+1])*(t**2)*(3 - 2*t) + (mk_list[0, i])*t*((1 - t)**2) + (mk_list[0, i+1])*(-(t**2)*(1-t))
            Hrmt[1, i*T.size + t] = (Polygon[1, i])*((1 - t)**2)*(1 + 2*t) + (Polygon[1, i+1])*(t**2)*(3 - 2*t) + (mk_list[1, i])*t*((1 - t)**2) + (mk_list[1, i+1])*(-(t**2)*(1-t))

    return Hrmt




class Index(object):

    def addPoint(self, event):
        Poly = AcquisitionNvxPoints(minmax,'or',':r')
        PlotHermiteCurve(Poly)
        
    def removePoint(self, event):
        Poly = AcquisitionRMVPoints(minmax,'or',':r')
        PlotHermiteCurve(Poly)


callback = Index()
axaddpoints = plt.axes([0.81, 0.05, 0.2, 0.075])
axrempoint = plt.axes([0.6, 0.05, 0.2, 0.075])
baddpoints = Button(axaddpoints, 'add Point')
baddpoints.on_clicked(callback.addPoint)
brempoints = Button(axrempoint, 'remove Point')
brempoints.on_clicked(callback.removePoint)
plt.show()

