import sys
import numpy as np
import qiskit
import matplotlib.pyplot as plt

from math import pi

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile

from qiskit.quantum_info import state_fidelity

from qiskit_aer import Aer
import qiskit_aer
from scipy.linalg import fractional_matrix_power
import time
plt.rcParams.update({'text.usetex': False})


start_time = time.time()

# %%
# Bath #
M: int = 60
xi: int = 2
wc: float = 1.5
beta: int = 1
h: int = 1

c = np.zeros(M)
w = np.zeros(M)
x = np.zeros(M)
p = np.zeros(M)
m = np.ones(M)
a = np.zeros(M)

rng = np.random.default_rng(seed=123456)

for i in range(M):
    j = i + 1  
    w[i] = wc * np.log(M / (M - j + .5))  
    c[i] = w[i] * np.sqrt(xi * m[i] * h * wc) / np.sqrt(M)


    x[i] = rng.normal(c[i]/(w[i]**2), 1/np.sqrt(beta * w[i]))
    p[i] = rng.normal(0, np.sqrt(w[i]/beta))

    # x[i] = rng.normal(0, 1/np.sqrt(2 * a[i] * w[i]))
    # p[i] = rng.normal(0, np.sqrt(w[i]/(2.0 * a[i])))
    
del h

# %%
# System #
e: float = 0.
hx: float = 1.
Psi0: list[complex] = np.array([1, 0], dtype='complex128')

# %%

ti: float = 0.
tf: float = 8
N: int = 650
dt: float = (tf - ti) / N
N += 1
t = np.linspace(ti, tf, N)

# %%



def H(t: float) -> list[list[complex]]:
    """
    Time-dependent Hamiltonian.

    Parameters
    ----------
    t : float
        Time.

    Returns
    -------
    list[list[complex]]
        Hamiltonian.

    """
    hzt = hz(t, e, x, p, c, w)
    return np.array([[hzt, hx], [hx, -hzt]], dtype='complex128')


def hz(t: float, e: float, x: list[float], p: list[float],
       c: list[float], w: list[float]) -> float:
    """
    Calculate coefficient of Pauli Z in Hamiltonian.

    Parameters
    ----------
    t : float
        Time.
    e : float
        Additive constant.
    x : list[float]
        Position.
    p : list[float]
        Momentum.
    c : list[float]
        Strength coefficient.
    w : list[float]
        Angular frequency.

    Returns
    -------
    float
        DESCRIPTION.

    """
    sho = 0
    for i in range(len(x)):
        sho += c[i] * x[i] * np.cos(w[i] * t)
        sho += c[i] * p[i] * np.sin(w[i] * t) / w[i]
    return sho + e
    # return 0.


###################################################################################################
#############################################  RK  ################################################
###################################################################################################

def RK4(f, t: list[float], h: float, y0: list[float]) -> list[list[float]]:
    """Real RK4."""
    y = np.zeros((len(t), len(y0)), dtype='float64')
    y[0] = y0
    n = 0
    while n < len(t) - 1:
        tn = t[n]
        yn = y[n]
        k1 = h * f(tn, yn)
        k2 = h * f(tn + h/2, yn + k1/2)
        k3 = h * f(tn + h/2, yn + k2/2)
        k4 = h * f(tn + h, yn + k3)
        y[n+1] = yn + (k1 + 2*k2 + 2*k3 + k4)/6
        n += 1
    return np.transpose(y)



def RK8(f, t: list[float], h: float, y0: list[float]) -> list[list[float]]:
    """Real RK8."""
    y = np.zeros((len(t), len(y0)), dtype='float64')
    y[0] = y0
    n: int = 0
    while n < len(t) - 1:
        tn = t[n]
        yn = y[n]
        s21 = np.sqrt(21)
        k1 = h * f(tn, yn)
        k2 = h * f(tn + h/2, yn + k1/2)
        k3 = h * f(tn + h/2, yn + k1/4 + k2/4)
        k4 = h * f(tn + h*(7 + s21)/14,
                   yn + k1/7 + k2*(-7 - 3*s21)/98 + k3*(21 + 5*s21)/49)
        k5 = h * f(tn + h*(7 + s21)/14,
                   yn + k1*(11 + s21)/84 + k3*(18 + 4*s21)/63 +
                   k4*(21 - s21)/252)
        k6 = h * f(tn + h/2,
                   yn + k1*(5 + s21)/48 + k3*(9 + s21)/36 +
                   k4*(-231 + 14*s21)/360 + k5*(63 - 7*s21)/80)
        k7 = h * f(tn + h*(7 - s21)/14,
                   yn + k1*(10 - s21)/42 + k3*(-432 + 92*s21)/315 +
                   k4*(633 - 145*s21)/90 + k5*(-504 + 115*s21)/70 +
                   k6*(63 - 13*s21)/35)
        k8 = h * f(tn + h*(7 - s21)/14,
                   yn + k1/14 + k5*(14 - 3*s21)/126 +
                   k6*(13 - 3*s21)/63 + k7/9)
        k9 = h * f(tn + h/2,
                   yn + k1/32 + k5*(91 - 21*s21)/576 +
                   k6*(11/72) + k7*(-385 - 75*s21)/1152 +
                   k8*(63 + 13*s21)/128)
        k10 = h * f(tn + h*(7 + s21)/14,
                    yn + k1/14 + k5/9 + k6*(-733 - 147*s21)/2205 +
                    k7*(515 + 111*s21)/504 + k8*(-51 - 11*s21)/56 +
                    k9*(132 + 28*s21)/245)
        k11 = h * f(tn + h,
                    yn + k5*(-42 + 7*s21)/18 + k6*(-18 + 28*s21)/45 +
                    k7*(-273 - 53*s21)/72 + k8*(301 + 53*s21)/72 +
                    k9*(28 - 28*s21)/45 + k10*(49 - 7*s21)/18)
        y[n+1] = yn + (9*k1 + 49*k8 + 64*k9 + 49*k10 + 9*k11) / 180
        n += 1
    return np.transpose(y)



def cRK4(f, t: list[float], h: float,
         z0: list[complex]) -> list[list[complex]]:
    """Complex RK4."""
    z = np.zeros((len(t), len(z0)), dtype='float64')
    z[0] = z0
    n: int = 0
    while n < len(t) - 1:
        tn = t[n]
        zn = z[n]
        k1 = h * f(tn, zn)
        k2 = h * f(tn + h/2, zn + h*k1/2)
        k3 = h * f(tn + h/2, zn + h*k2/2)
        k4 = h * f(tn + h, zn + h*k3)
        z[n+1] = zn + (k1 + 2*k2 + 2*k3 + k4)/6
        n += 1
    return np.transpose(z)



def cRK8(f, t: list[float], h: float,
         z0: list[complex]) -> list[list[complex]]:
    """Complex RK8."""
    z = np.zeros((len(t), len(z0)), dtype='complex128')
    z[0] = z0
    n: int = 0
    while n < len(t) - 1:
        tn = t[n]
        zn = z[n]
        s21 = np.sqrt(21)
        k1 = h * f(tn, zn)
        k2 = h * f(tn + h/2, zn + k1/2)
        k3 = h * f(tn + h/2, zn + k1/4 + k2/4)
        k4 = h * f(tn + h*(7 + s21)/14,
                   zn + k1/7 + k2*(-7 - 3*s21)/98 + k3*(21 + 5*s21)/49)
        k5 = h * f(tn + h*(7 + s21)/14,
                   zn + k1*(11 + s21)/84 + k3*(18 + 4*s21)/63 +
                   k4*(21 - s21)/252)
        k6 = h * f(tn + h/2,
                   zn + k1*(5 + s21)/48 + k3*(9 + s21)/36 +
                   k4*(-231 + 14*s21)/360 + k5*(63 - 7*s21)/80)
        k7 = h * f(tn + h*(7 - s21)/14,
                   zn + k1*(10 - s21)/42 + k3*(-432 + 92*s21)/315 +
                   k4*(633 - 145*s21)/90 + k5*(-504 + 115*s21)/70 +
                   k6*(63 - 13*s21)/35)
        k8 = h * f(tn + h*(7 - s21)/14,
                   zn + k1/14 + k5*(14 - 3*s21)/126 +
                   k6*(13 - 3*s21)/63 + k7/9)
        k9 = h * f(tn + h/2,
                   zn + k1/32 + k5*(91 - 21*s21)/576 +
                   k6*(11/72) + k7*(-385 - 75*s21)/1152 +
                   k8*(63 + 13*s21)/128)
        k10 = h * f(tn + h*(7 + s21)/14,
                    zn + k1/14 + k5/9 + k6*(-733 - 147*s21)/2205 +
                    k7*(515 + 111*s21)/504 + k8*(-51 - 11*s21)/56 +
                    k9*(132 + 28*s21)/245)
        k11 = h * f(tn + h,
                    zn + k5*(-42 + 7*s21)/18 + k6*(-18 + 28*s21)/45 +
                    k7*(-273 - 53*s21)/72 + k8*(301 + 53*s21)/72 +
                    k9*(28 - 28*s21)/45 + k10*(49 - 7*s21)/18)
        z[n+1] = zn + (9*k1 + 49*k8 + 64*k9 + 49*k10 + 9*k11) / 180
        n += 1
    return np.transpose(z)

# %%



def Schrodinger(t: float, Psi: list[complex]) -> list[complex]:
    """
    Get time-derivative of wave function.

    Parameters
    ----------
    t : float
        Time.
    Psi : list[complex]
        Wave function.

    Returns
    -------
    list[complex]
        Time-derivative of wave function.

    """
    return -1j * H(t) @ Psi


num = 100001
h = (tf - ti)/num
timepsi = np.linspace(ti, tf, num)


Psi = cRK8(Schrodinger, timepsi, h, Psi0)


###################################################################################################
#########################################  GradDescent  ###########################################
###################################################################################################

sim = Aer.get_backend("qasm_simulator")
shots = 50000

def getL(thetaold, dtheta, t, dt):
    hx = 1
    hz1 = hz(t, e, x, p, c, w)

    thetanew = thetaold + dtheta
    
    q = QuantumRegister(1)
    c1 = ClassicalRegister(1) 
    qc = QuantumCircuit(q, c1) 
    
    qc.rz(thetanew[0], q)
    qc.ry(thetanew[1], q)
    qc.rz(thetanew[2], q)
 
    qc.rx(hx*dt, q)
    qc.rz(-2*hz1*dt, q)
    qc.rx(hx*dt, q)

    qc.rz(-1* thetaold[2], q)
    qc.ry(-1* thetaold[1], q)
    qc.rz(-1* thetaold[0], q)

    qc.measure(0, 0)
    
    circuitTranspile = transpile(qc, sim)

    run = sim.run(circuitTranspile, shots= shots).result().get_counts(0)
    L = (1 - (run.get('0',0)/shots))/(dt*dt)
    
    return L


def gradient(thetaold, dtheta, t, dt, s):
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    grad1 = (getL(thetaold, dtheta + (s*e1), t, dt) - getL(thetaold, dtheta - (s*e1), t, dt))/(2*np.sin(s))

    grad2 = (getL(thetaold, dtheta + (s*e2), t, dt) - getL(thetaold, dtheta - (s*e2), t, dt))/(2*np.sin(s))

    grad3 = (getL(thetaold, dtheta + (s*e3), t, dt) - getL(thetaold, dtheta - (s*e3), t, dt))/(2*np.sin(s))

    gradArray = np.array([grad1, grad2, grad3])
    return gradArray


def newdtheta(dthetaold, theta, t, dt, s, eta):
    
    return dthetaold - eta*gradient(theta, dthetaold, t, dt, s)

def measure(theta):
    
    q = QuantumRegister(1) 
    c = ClassicalRegister(1)

    qc = QuantumCircuit(q, c)
    qc.rz(theta[0], q)
    qc.ry(theta[1], q)
    qc.rz(theta[2], q)

    qc.measure(0,0)

    circuitTranspile = transpile(qc, sim)

    run = sim.run(circuitTranspile, shots= shots).result().get_counts(0)

    return np.array([run.get('0',0)/shots, run.get('1',0)/shots])

# %%
# Minimization #

dt = .01
s = np.pi/2
eta = 0.005
epsilon = 0.001

dtheta = np.array([.01,.01,.01])
theta = np.array([0.0,0.0,0.0])

thetas = []

timesum = 0
timeAxis = []
thetas0 = []
thetas1 = []

alpha =  0.001
beta1 = 0.9
beta2 = 0.999
ep = 10**-8

for i in range(0, 800):
    StepsInMin = 0
    m0 = np.zeros(3)
    v0 = np.zeros(3) 
    timesum += dt
    for j in range(0,500):
        dthetaTEMP = dtheta

        grad = gradient(theta, dthetaTEMP, timesum, dt, s)
        m0 = beta1 * m0 + (1 - beta1) * grad
        v0 = beta2 * v0 + (1 - beta2) * np.square(grad)

        mhat = m0/(1 - beta1**(j+1))
        vhat = v0/(1 - beta2**(j+1))

        dtheta = dthetaTEMP - alpha * (mhat/(np.sqrt(vhat) + ep*np.ones(3)))

        StepsInMin += 1
        L = getL(theta, dtheta, timesum, dt)

        print(L)
        
        if(L < 0.000001):
            break
    
    theta += dtheta
    thetas.append(theta)

    thetasx = measure(thetas[i])
    thetas0.append(thetasx[0])
    thetas1.append(thetasx[1])

    timeAxis.append(timesum)
    

    print(i)
    print("##########")


print("--- %s seconds ---" % (time.time() - start_time))


plt.figure(dpi=250, layout='constrained')
plt.plot(timepsi, np.abs(Psi[0])**2, label=r'RK8', color = "black")
plt.plot(timeAxis, thetas0,label=r'Adam'
         "\n"
         r"$\beta_1 = 0.9,  \beta_2 = 0.999$", color = 'red')

plt.title(r"Time-Evolution of Wave Function (Seed: 123456)")
plt.xlabel(f'Time [dt = {h:.2e}]')
plt.ylabel(r'$\Psi(t) = [\psi(t) \quad \phi(t)]^\mathrm{T}$')
plt.legend(ncol=2)
plt.savefig("VQDPlot", dpi=300)
plt.show()
del num, h
