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


def Norm(vect):
    return vect[0]**2 + vect[1]**2


def WhatAreMoMn(Polygon,c):
    # Cas k
    if Polygon.shape[1] <= 1:
        return np.array((1,1)),np.array((1,1))
    if Polygon.shape[1] == 2:
        m0 = Polygon[:,1] - Polygon[:,0]
        mn = Polygon[:,1] - Polygon[:,0]
        return m0, mn
    choice = input('Voulez-vous choisir m0 et mn ? (Non/oui) : ')
    P0 = Polygon[:,0]
    Pn = Polygon[:,-1]
    if choice.lower() == 'oui':
        # Cas où on choisit m0 et mn
        print("Attention !!! Vos deux prochains clics sont importants !!!")
        print("Votre 1er lieu de clic sera le bout d'un vecteur partant de votre P0")
        print("Ce vecteur sera la tangente en P0 de la courbe")
        print("Votre second clic fera presque la même chose mais en Pn")
        coord = []
        while coord == []:
            print("Merci de rentrer m0")
            coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        m0x = coord[0][0]
        m0y = coord[0][1]
        coord = []
        while coord == []:
            print("Merci de rentrer mn")
            coord = plt.ginput(1, mouse_add=1, mouse_stop=3, mouse_pop=2)
        mnx = coord[0][0]
        mny = coord[0][1]
        return np.array((m0x,m0y))-P0, np.array((mnx,mny))-Pn
    # Cas automatique
    P1 = Polygon[:,1]
    P2 = Polygon[:,2]
    Pnm1 = Polygon[:,-2]
    Pnm2 = Polygon[:,-3]
    m1 = (1-c) / 2 * (P2 - P0)
    mnm1 = (1-c) / 2 * (Pn - Pnm2)
    Pt0 = P1 - m1
    Ptn = Pnm1 + mnm1
    while Norm(Pt0-P0) > Norm(m1):
        Pt0 -= 0.01 * (P1-P0)
    while Norm(Ptn-Pn) > Norm(mnm1):
        Ptn += 0.01 * (Pn - Pnm1)
    return Pt0-P0, Pn-Ptn


def PlotHermiteCurve(Polygon):
    c = WhatIsC()
    m0, mn = WhatAreMoMn(Polygon, c)

    N = len(Polygon[0, :])-1
    t = np.linspace(0,1,500)

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
        Liste[1, i] = (1-c)*(Polygon[1, i+1] - Polygon[1, i-1])/2
    return Liste


def Hermite(N, T, m0, mn, c, Polygon):
    
    mk_list = Mk_List(N, m0, mn, c, Polygon)
    Hrmt = np.zeros((2, N*T.size))

    for i in range(N):
        for k in range (T.size):
            
            Hrmt[0, i*T.size + k] =  (Polygon[0, i])*((1 - T[k])**2)*(1 + 2*T[k]) + (Polygon[0, i+1])*(T[k]**2)*(3 - 2*T[k]) + (mk_list[0, i])*T[k]*((1 - T[k])**2) + (mk_list[0, i+1])*(-(T[k]**2)*(1-T[k]))
            Hrmt[1, i*T.size + k] =  (Polygon[1, i])*((1 - T[k])**2)*(1 + 2*T[k]) + (Polygon[1, i+1])*(T[k]**2)*(3 - 2*T[k]) + (mk_list[1, i])*T[k]*((1 - T[k])**2) + (mk_list[1, i+1])*(-(T[k]**2)*(1-T[k]))

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

