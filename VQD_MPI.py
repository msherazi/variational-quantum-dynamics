import sys, gc 
import pickle
from time import process_time

import numpy as np
from scipy import linalg
from numba import njit
#import matplotlib.pyplot as plt

from mpi4py import MPI

from qiskit import QuantumCircuit,ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit import Parameter

from qiskit_aer import AerSimulator

from qiskit_aer.noise import NoiseModel

# %%
# Bath #
ICsPerProc = 20
MC_Blocks = 10
N_Osc: int = 60
xi: float = 1.2
wc: float = 2.5
beta: float = 0.2

# %%
# System #
e: float = 0.
hx: float = 1.
Psi0: list[complex] = np.array([1, 0], dtype='complex128')

# %%
# Simulation #
# OutFile_BaseName = 'Test_Dt_.025_MC_20'
OutFile_BaseName = 'Xi_1.2_Wc_2.5_Beta_0.2_MC_10k'
# OutFile_BaseName = 'Xi_2_Wc_1.5_Beta_1_Test'
Dt:         float = 2.5e-2  #Ehrenfest dt
StepsPerDt: int = 25      #Integrator steps per Ehrenfest Dt/StepsPerDt shuld be less than 1e-3 
ti:         float = 0.
tf:         float = 10.
EhrenfestOff: bool = False  #Can be used to turn off Ehrenfest
seed = 123456 

# %%
# VQD #
dtheta0 =  np.array([.01,.01,.01])
theta0 =   np.array([0.0,0.0,0.0])
Shots:    int = 50000
OptAngle: float = np.pi/2
NoQuantum: bool = False    #Can be used to quit the coed before running quantum parts

# %%
# Minimization #
alpha:   float = 0.01
beta1:   float = 0.9
beta2:   float = 0.999
epsilon: float = 10**-8
MaxSteps: int = 500
L_Cutoff: float = 1e-6

# %%
Comm = MPI.COMM_WORLD
Rank = Comm.Get_rank()
NProc = Comm.Get_size()
# %%

hb: int = 1

ProcsPerBlock = NProc / MC_Blocks

if (ProcsPerBlock-int(ProcsPerBlock)) != 0.0:
    print("The # of MC_Blocks is not an integer multiple of the number of ICs.")
    sys.exit(1)   
else:
    ProcsPerBlock = int(ProcsPerBlock)

TimeSteps_Dt = (tf - ti)/Dt
if (TimeSteps_Dt-int(TimeSteps_Dt)) != 0.0:
    print("Dt is not an integer multiple of the number of the time span.")
    sys.exit(1)   
else:
   TimeSteps_Dt = int(TimeSteps_Dt)

TimePoints_Dt = TimeSteps_Dt + 1

TimeSteps = TimeSteps_Dt * StepsPerDt
TimePoints = TimeSteps + 1
dt =  (tf - ti)/TimeSteps


Time_dt = np.linspace(ti, tf, TimePoints)
Time_Dt = np.linspace(ti, tf, TimePoints_Dt)

# %%
Pauli_X = np.array([[0,1], [1, 0]], dtype='complex128')
Pauli_Y = np.array([[0,-1j], [1j, 0]], dtype='complex128')
Pauli_Z = np.array([[1,0], [0, -1]], dtype='complex128')

# %%

backend_brisbane = None

if Rank == 0:
    with open('backend_brisbane_012524_expanse.pkl', 'rb') as f:
       backend_brisbane  = pickle.load(f)

backend_brisbane = Comm.bcast(backend_brisbane, root=0)

noise_model = NoiseModel.from_backend(backend_brisbane)

# # Get coupling map from backend
coupling_map = backend_brisbane.configuration().coupling_map

# # Get basis gates from noise model
basis_gates = noise_model.basis_gates

NoisyBackend_Name = 'NoisyBrisbane'
backend_noisy = AerSimulator(noise_model=noise_model,coupling_map=coupling_map,basis_gates=basis_gates,max_parallel_threads=1)
backend = AerSimulator(max_parallel_threads=1)

# %%
@njit
def GetDMatFromPsi(Psi: list[complex]):

    Rho_00 = Psi[0]*np.conj(Psi[0])
    Rho_01 = Psi[0]*np.conj(Psi[1])
    Rho_10 = Psi[1]*np.conj(Psi[0])
    Rho_11 = Psi[1]*np.conj(Psi[1])

    return np.array([[Rho_00,Rho_01], [Rho_10, Rho_11]], dtype='complex128')

@njit
def GetExptS(DMat: np.ndarray):
    if EhrenfestOff == True:
        return 0
    return np.real(np.trace(DMat @ Pauli_Z))

@njit
def GetNewXandP(x0: list[float], p0: list[float], 
                s: float, t: float) -> tuple[list[float],list[float]] :

    xNew = np.zeros(len(x0), dtype='float64')
    pNew = np.zeros(len(p0), dtype='float64')

    for i in range(len(x0)):
        xNew[i] += x0[i] * np.cos(w[i] * t)
        xNew[i] += p0[i] * np.sin(w[i] * t) / w[i]
        xNew[i] += ((s * c[i])/(2.0 * w[i])) * ((1 - np.cos(w[i] * t))/w[i])

        pNew[i] += p0[i] * np.cos(w[i] * t) 
        pNew[i] -= x0[i] * w[i] * np.sin(w[i] * t)
        pNew[i] += (s * c[i])/(2.0 * w[i]) * np.sin(w[i] * t)
    
    return (xNew,pNew)

@njit
def H(x: list[float], p: list[float], s: float, t: float) -> list[list[complex]]:
    """
    Time-dependent Hamiltonian.

    Parameters
    ----------
    x : list[float]
        Position.
    p : list[float]
        Momentum.
    s : float
        Reference system coordinate
    t : float
        Time.
    Returns
    -------
    list[list[complex]]
        Hamiltonian.

    """
    hzt = hz(t, e, x, p, s, c, w)
    return np.array([[hzt, hx], [hx, -hzt]], dtype='complex128')


@njit
def hz(t: float, e: float, x: list[float], p: list[float],
       s: float, c: list[float], w: list[float]) -> float:
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
    s : float
        Reference system coordinate        
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
        sho += ((s * c[i]**2)/(2.0 * w[i])) * ((1 - np.cos(w[i] * t))/w[i])
    return e - sho

@njit
def RK4(f, t: list[float], h: float, y0: list[float],args: tuple) -> list[list[float]]:
    """Real RK4."""
    y = np.zeros((len(t), len(y0)), dtype='float64')
    y[0] = y0
    n = 0
    while n < len(t) - 1:
        tn = t[n]
        yn = y[n]
        k1 = h * f(args, tn, yn)
        k2 = h * f(args, tn + h/2, yn + k1/2)
        k3 = h * f(args, tn + h/2, yn + k2/2)
        k4 = h * f(args, tn + h, yn + k3)
        y[n+1] = yn + (k1 + 2*k2 + 2*k3 + k4)/6
        n += 1
    return np.transpose(y)


@njit
def RK8(f, t: list[float], h: float, y0: list[float], args: tuple) -> list[list[float]]:
    """Real RK8."""
    y = np.zeros((len(t), len(y0)), dtype='float64')
    y[0] = y0
    n: int = 0
    while n < len(t) - 1:
        tn = t[n]
        yn = y[n]
        s21 = np.sqrt(21)
        k1 = h * f(args, tn, yn)
        k2 = h * f(args, tn + h/2, yn + k1/2)
        k3 = h * f(args, tn + h/2, yn + k1/4 + k2/4)
        k4 = h * f(args, tn + h*(7 + s21)/14,
                   yn + k1/7 + k2*(-7 - 3*s21)/98 + k3*(21 + 5*s21)/49)
        k5 = h * f(args, tn + h*(7 + s21)/14,
                   yn + k1*(11 + s21)/84 + k3*(18 + 4*s21)/63 +
                   k4*(21 - s21)/252)
        k6 = h * f(args, tn + h/2,
                   yn + k1*(5 + s21)/48 + k3*(9 + s21)/36 +
                   k4*(-231 + 14*s21)/360 + k5*(63 - 7*s21)/80)
        k7 = h * f(args, tn + h*(7 - s21)/14,
                   yn + k1*(10 - s21)/42 + k3*(-432 + 92*s21)/315 +
                   k4*(633 - 145*s21)/90 + k5*(-504 + 115*s21)/70 +
                   k6*(63 - 13*s21)/35)
        k8 = h * f(args, tn + h*(7 - s21)/14,
                   yn + k1/14 + k5*(14 - 3*s21)/126 +
                   k6*(13 - 3*s21)/63 + k7/9)
        k9 = h * f(args, tn + h/2,
                   yn + k1/32 + k5*(91 - 21*s21)/576 +
                   k6*(11/72) + k7*(-385 - 75*s21)/1152 +
                   k8*(63 + 13*s21)/128)
        k10 = h * f(args, tn + h*(7 + s21)/14,
                    yn + k1/14 + k5/9 + k6*(-733 - 147*s21)/2205 +
                    k7*(515 + 111*s21)/504 + k8*(-51 - 11*s21)/56 +
                    k9*(132 + 28*s21)/245)
        k11 = h * f(args, tn + h,
                    yn + k5*(-42 + 7*s21)/18 + k6*(-18 + 28*s21)/45 +
                    k7*(-273 - 53*s21)/72 + k8*(301 + 53*s21)/72 +
                    k9*(28 - 28*s21)/45 + k10*(49 - 7*s21)/18)
        y[n+1] = yn + (9*k1 + 49*k8 + 64*k9 + 49*k10 + 9*k11) / 180
        n += 1
    return np.transpose(y)


@njit
def cRK4(f, t: list[float], h: float,
         z0: list[complex], args: tuple) -> list[list[complex]]:
    """Complex RK4."""
    z = np.zeros((len(t), len(z0)), dtype='float64')
    z[0] = z0
    n: int = 0
    while n < len(t) - 1:
        tn = t[n]
        zn = z[n]
        k1 = h * f(args, tn, zn)
        k2 = h * f(args, tn + h/2, zn + h*k1/2)
        k3 = h * f(args, tn + h/2, zn + h*k2/2)
        k4 = h * f(args, tn + h, zn + h*k3)
        z[n+1] = zn + (k1 + 2*k2 + 2*k3 + k4)/6
        n += 1
    return np.transpose(z)


@njit
def cRK8(f, t: list[float], h: float,
         z0: list[complex], args: tuple) -> list[list[complex]]:
    """Complex RK8."""
    z = np.zeros((len(t), len(z0)), dtype='complex128')
    z[0] = z0
    n: int = 0
    while n < len(t) - 1:
        tn = t[n]
        zn = z[n]
        s21 = np.sqrt(21)
        k1 = h * f(args, tn, zn)
        k2 = h * f(args, tn + h/2, zn + k1/2)
        k3 = h * f(args, tn + h/2, zn + k1/4 + k2/4)
        k4 = h * f(args, tn + h*(7 + s21)/14,
                   zn + k1/7 + k2*(-7 - 3*s21)/98 + k3*(21 + 5*s21)/49)
        k5 = h * f(args, tn + h*(7 + s21)/14,
                   zn + k1*(11 + s21)/84 + k3*(18 + 4*s21)/63 +
                   k4*(21 - s21)/252)
        k6 = h * f(args, tn + h/2,
                   zn + k1*(5 + s21)/48 + k3*(9 + s21)/36 +
                   k4*(-231 + 14*s21)/360 + k5*(63 - 7*s21)/80)
        k7 = h * f(args, tn + h*(7 - s21)/14,
                   zn + k1*(10 - s21)/42 + k3*(-432 + 92*s21)/315 +
                   k4*(633 - 145*s21)/90 + k5*(-504 + 115*s21)/70 +
                   k6*(63 - 13*s21)/35)
        k8 = h * f(args, tn + h*(7 - s21)/14,
                   zn + k1/14 + k5*(14 - 3*s21)/126 +
                   k6*(13 - 3*s21)/63 + k7/9)
        k9 = h * f(args, tn + h/2,
                   zn + k1/32 + k5*(91 - 21*s21)/576 +
                   k6*(11/72) + k7*(-385 - 75*s21)/1152 +
                   k8*(63 + 13*s21)/128)
        k10 = h * f(args, tn + h*(7 + s21)/14,
                    zn + k1/14 + k5/9 + k6*(-733 - 147*s21)/2205 +
                    k7*(515 + 111*s21)/504 + k8*(-51 - 11*s21)/56 +
                    k9*(132 + 28*s21)/245)
        k11 = h * f(args, tn + h,
                    zn + k5*(-42 + 7*s21)/18 + k6*(-18 + 28*s21)/45 +
                    k7*(-273 - 53*s21)/72 + k8*(301 + 53*s21)/72 +
                    k9*(28 - 28*s21)/45 + k10*(49 - 7*s21)/18)
        z[n+1] = zn + (9*k1 + 49*k8 + 64*k9 + 49*k10 + 9*k11) / 180
        n += 1
    return np.transpose(z)

# %%


@njit
def Schrodinger(args: tuple[list[float],list[float],float],
                t: float, Psi: list[complex]) -> list[complex]:
    """
    Get time-derivative of wave function.

    Parameters
    ----------
    args: tuple[ 
        x : list[float]
            Position.
        p : list[float]
            Momentum.
        s : float
            Reference system coordinate
        ]
    t : float
        Time.
    Psi : list[complex]
        Wave function.

    Returns
    -------
    list[complex]
        Time-derivative of wave function.

    """

    x = args[0]
    p = args[1]
    s = args[2]

    return -1j * H(x,p,s,t) @ Psi

# %%
@njit
def GetPropagater_TwoStepSymTrotter(Dt: float, hx: float, 
                                    hz_0: float, hz_1: float) -> np.ndarray:
    
    M_00  = np.exp(-1j * (1.0/2.0) * Dt * (hz_0 + hz_1))
    M_00 *= np.cos(Dt * hx)

    M_01  = -1j*np.exp(1j * (1.0/2.0) * Dt * (hz_0 - hz_1))
    M_01 *= np.sin(Dt * hx) 

    M_10  = -1j*np.exp(-1j * (1.0/2.0) * Dt * (hz_0 - hz_1))
    M_10 *= np.sin(Dt * hx) 

    M_11  = np.exp(1j * (1.0/2.0) * Dt * (hz_0 + hz_1))
    M_11 *= np.cos(Dt * hx)


    M = np.array([[M_00,M_01], [M_10, M_11]])

    return M

# %%
def getL_Sim(backend, thetaold, dtheta, Dt, hzVals):

    hz_0 = hzVals[0]
    hz_1 = hzVals[1]

    thetanew = thetaold + dtheta
    
    q = QuantumRegister(1) # creates a Quantum register of 1 qubit
    c1 = ClassicalRegister(1) # creates a classical register of 1 bit
    qc = QuantumCircuit(q, c1) # creates a quantum circuit 
    
    qc.rz(thetanew[0], q)
    qc.ry(thetanew[1], q)
    qc.rz(thetanew[2], q)
 
    qc.rz(-hz_1*Dt, q)
    qc.rx(-2*hx*Dt, q)
    qc.rz(-hz_0*Dt, q)

    qc.rz(-1* thetaold[2], q)
    qc.ry(-1* thetaold[1], q)
    qc.rz(-1* thetaold[0], q)

    qc.measure(0, 0)
    
    circuitTranspile = transpile(qc, backend)

    run = backend.run(circuitTranspile, shots= Shots).result().get_counts(0)
    L = (1 - (run.get('0',0)/Shots))/(Dt*Dt)
   # print (L)
    return L #run [1] for p1

# %%
def getL_Analytic(thetaold, dtheta, Dt, hzVals):

    hz_0 = hzVals[0]
    hz_1 = hzVals[1]

    thetanew = thetaold + dtheta
    
    # All terms computed analytically using Mathematica

    A_exp  = Dt * (hz_0 + hz_1) 
    A_exp += thetanew[0] + thetanew[2]
    A_exp += thetaold[2] - thetaold[0]
    A = np.exp(-1j * (1.0/2.0) * A_exp)

    B_1  = np.exp(1j * thetanew[2]) * np.sin(thetanew[1] / 2.0)
    B_2  = np.exp(1j * (Dt * hz_0 + thetaold[2]))
    B_2 *= np.cos(thetaold[1] / 2.0) * np.sin(Dt * hx)
    B_3  = np.sin(thetaold[1] / 2.0) * np.cos(Dt * hx)
    B = B_1 * (1j *B_2 + B_3) 

    C_1  = np.exp(1j * Dt * hz_1) * np.cos(thetanew[1] / 2.0)
    C_2  = np.exp(1j * (Dt * hz_0 + thetaold[2]))
    C_2 *= np.cos(thetaold[1] / 2.0) * np.cos(Dt * hx)
    C_3  = np.sin(thetaold[1] / 2.0) * np.sin(Dt * hx)
    C = C_1 * (C_2 + 1j * C_3)

    Res = A * (B + C) 

    L = (1 - np.abs(Res)**2)/(Dt*Dt)
   # print (L)
    return L #run [1] for p1

# %%
def getL(backend, thetaold, dtheta, Dt, hzVals):
    if backend == None:
        return getL_Analytic(thetaold, dtheta, Dt, hzVals)
    
    return getL_Sim(backend, thetaold, dtheta, Dt, hzVals)

# %%
def gradient(backend, thetaold, dtheta, Dt, hzVals):
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    grad1  = getL(backend, thetaold, dtheta + (OptAngle*e1), Dt, hzVals)
    grad1 -= getL(backend, thetaold, dtheta - (OptAngle*e1), Dt, hzVals)        
    grad1 /= 2*np.sin(OptAngle)

    grad2  = getL(backend, thetaold, dtheta + (OptAngle*e2), Dt, hzVals)
    grad2 -= getL(backend, thetaold, dtheta - (OptAngle*e2), Dt, hzVals)        
    grad2 /= 2*np.sin(OptAngle)

    grad3  = getL(backend, thetaold, dtheta + (OptAngle*e3), Dt, hzVals)
    grad3 -= getL(backend, thetaold, dtheta - (OptAngle*e3), Dt, hzVals)        
    grad3 /= 2*np.sin(OptAngle)  

    gradArray = np.array([grad1, grad2, grad3])
    return gradArray

# %%
@njit
def measure(theta: list[float], Psi0: list[complex]) -> list[complex]:
    
    M_00 = np.exp(-1j * (1.0/2.0) * (theta[0] + theta[2]))
    M_00 *= np.cos(theta[1] / 2.0)

    M_01 = -np.exp(1j * (1.0/2.0) * (theta[0] - theta[2]))
    M_01 *= np.sin(theta[1] / 2.0) 

    M_10 = np.exp(-1j * (1.0/2.0) * (theta[0] - theta[2]))
    M_10 *= np.sin(theta[1] / 2.0) 

    M_11 = np.exp(1j * (1.0/2.0) * (theta[0] + theta[2]))
    M_11 *= np.cos(theta[1] / 2.0)

    M = np.array([[M_00,M_01], [M_10, M_11]])

    return np.dot(M,Psi0)

# %%
@njit
def Ehrenfest(Integrator, Dt: float, dt: float, Psi0: list[complex],
              args0: tuple[list[float],list[float],float]) -> list[list[complex]]:

    x0 = args0[0]
    p0 = args0[1]
    s0 = args0[2]

    time_in = np.linspace(0, Dt, StepsPerDt + 1)

    Psi = np.zeros((TimePoints, len(Psi0)), dtype='complex128')
    for ts_out in range(TimeSteps_Dt):
        Psi_Temp = Integrator(Schrodinger, time_in, dt, Psi0, (x0 ,p0 ,s0 ) )
        (x0,p0) = GetNewXandP(x0, p0, s0, Dt)
        Psi0 = Psi_Temp[:,-1]
        s0 = GetExptS(GetDMatFromPsi(Psi0))

        for ts_in in range(StepsPerDt):
            Psi[ts_out * StepsPerDt + ts_in] = Psi_Temp[:,ts_in]

    Psi[-1] = Psi_Temp[:,-1]  # Account for last time point

    return np.transpose(Psi) 

# %%
@njit
def Ehrenfest_Trotter(Dt: float, Psi0: list[complex],
                      args0: tuple[list[float],list[float],float]) -> list[list[complex]]:

    x0 = args0[0]
    p0 = args0[1]
    s0 = args0[2]

    Psi = np.zeros((TimePoints_Dt, len(Psi0)), dtype='complex128')
    Psi[0] = Psi0

    for ts_out in range(TimeSteps_Dt):
        hz_0 = hz(0,e,x0,p0,s0,c,w)
        hz_1 = hz(Dt,e,x0,p0,s0,c,w)
        
        SymTrotter = GetPropagater_TwoStepSymTrotter(Dt,hx,hz_0,hz_1)

        Psi[ts_out + 1] = np.dot(SymTrotter,Psi[ts_out])

        (x0,p0) = GetNewXandP(x0, p0, s0, Dt)
        s0 = GetExptS(GetDMatFromPsi(Psi[ts_out + 1]))

    return np.transpose(Psi)

# %%
def Ehrenfest_VQD(backend, Dt: float, Psi0: list[complex],
                  args0: tuple[list[float],list[float],float]) -> tuple[int,list[list[complex]]]:

    x0 = args0[0]
    p0 = args0[1]
    s0 = args0[2]

    theta = np.array(theta0)
    dtheta = np.array(dtheta0)

    StepsTaken = np.zeros(TimePoints_Dt, dtype='int64')
    Psi = np.zeros((TimePoints_Dt, len(Psi0)), dtype='complex128')
    Psi[0] = Psi0

    for ts_out in range(TimeSteps_Dt):
        hz_0 = hz(0,e,x0,p0,s0,c,w)
        hz_1 = hz(Dt,e,x0,p0,s0,c,w)
        
        m0 = np.zeros(3)
        v0 = np.zeros(3)
        for j in range(MaxSteps):

            grad = gradient(backend, theta, dtheta,Dt,(hz_0,hz_1))
            m0 = beta1 * m0 + (1 - beta1) * grad
            v0 = beta2 * v0 + (1 - beta2) * np.square(grad)

            mhat = m0/(1 - beta1**(j+1))
            vhat = v0/(1 - beta2**(j+1))

            dtheta -= alpha * (mhat/(np.sqrt(vhat) + epsilon*np.ones(3)))

            L = getL(backend,theta, dtheta,Dt,(hz_0,hz_1))
            if(L < L_Cutoff):
                break

        theta += dtheta

        StepsTaken[ts_out + 1] = StepsTaken[ts_out] + j
        Psi[ts_out + 1] = measure(theta,Psi0)
        (x0,p0) = GetNewXandP(x0, p0, s0, Dt)
        s0 = GetExptS(GetDMatFromPsi(Psi[ts_out + 1]))

    return (StepsTaken, np.transpose(Psi))  

# %%

# zeros(length of array) -> np.array
c = np.zeros(N_Osc)
w = np.zeros(N_Osc)
a = np.zeros(N_Osc)
m = np.ones(N_Osc)
x = np.zeros((ICsPerProc,N_Osc))
p = np.zeros((ICsPerProc,N_Osc))

seed +=  Rank * 125
rng = np.random.default_rng(seed=seed)

s0 = GetExptS(GetDMatFromPsi(Psi0))

for i in range(N_Osc):
    j = i + 1 
    w[i] = wc * np.log(N_Osc / (N_Osc - j + .5)) 
    c[i] = w[i] * np.sqrt(xi * m[i] * hb * wc) / np.sqrt(N_Osc)
    a[i] = np.tanh((1.0/2.0) * w[i] * beta)
    for k in range(ICsPerProc):
        x[k][i] = rng.normal((s0 * c[i])/(w[i] * w[i]), 1/np.sqrt(2 * a[i] * w[i]))
        p[k][i] = rng.normal(0, np.sqrt(w[i] / (2.0 * a[i])))

# %%

###################################################################################################
#############################################  RK8  ###############################################                        
###################################################################################################

T_Start = process_time()

Rho_OnProc = np.zeros((TimePoints,2,2), dtype='complex128')
for ic in range(ICsPerProc):
    Psi = Ehrenfest(cRK8, Dt, dt, Psi0,(x[ic],p[ic],s0))
    for ts in range(TimePoints):
        Rho_OnProc[ts] +=  GetDMatFromPsi(Psi[:,ts])   
Rho_OnProc /= ICsPerProc

SendCounts_Rho = np.array(Comm.gather(Rho_OnProc.size, 0))

if Rank != 0:
    Comm.Gatherv(sendbuf=Rho_OnProc, recvbuf=None, root=0)

if Rank == 0:
    Rho_FullList = np.empty(sum(SendCounts_Rho), dtype=Rho_OnProc.dtype)
    Comm.Gatherv(sendbuf=Rho_OnProc, recvbuf=(Rho_FullList, SendCounts_Rho), root=0)
    Rho_FullList = Rho_FullList.reshape((MC_Blocks, ProcsPerBlock, TimePoints, 2, 2))
    Rho_BlockList = np.average(Rho_FullList, axis = 1)

    P00_BlockList = np.real(Rho_BlockList.transpose((2,3,0,1))[0][0])
    P11_BlockList = np.real(Rho_BlockList.transpose((2,3,0,1))[1][1])

    P00_Avg = np.average(P00_BlockList, axis = 0)
    P11_Avg = np.average(P11_BlockList, axis = 0)

    P00_Std = np.std(P00_BlockList, axis = 0, ddof=1) / np.sqrt(MC_Blocks)
    P11_Std = np.std(P11_BlockList, axis = 0, ddof=1) / np.sqrt(MC_Blocks)

    OutFileName = f"{OutFile_BaseName}_Direct_Ehrenfest.csv"
    ErrFileName = f"{OutFile_BaseName}_Direct_Ehrenfest-Err.csv"

    count = 0
    with open(f"{OutFileName}",'w') as f:
        for t in Time_dt:
            RhoString = f"{t:.10f}, {P00_Avg[count]:.10f}, {P11_Avg[count]:.10f}"
            print(f"{RhoString}",file=f)
            count += 1

    count = 0
    with open(f"{ErrFileName}",'w') as f:
        for t in Time_dt:
            RhoString = f"{t:.10f}, {P00_Std[count]:.10f}, {P11_Std[count]:.10f}"
            print(f"{RhoString}",file=f)
            count += 1


    T_End = process_time() 
    print(f"Finished Ehrenfest: Time = {T_End - T_Start}",flush=True)
    print(f"Time/IC = {(T_End - T_Start) / ICsPerProc}",flush=True)

    del Rho_FullList,Rho_BlockList,P00_BlockList,P11_BlockList
    del P00_Avg,P11_Avg,P00_Std,P11_Std

del Psi,Rho_OnProc,SendCounts_Rho 

gc.collect()

# %%

###################################################################################################
###########################################  Trotter  #############################################                        
###################################################################################################

T_Start = process_time()

Rho_OnProc = np.zeros((TimePoints_Dt,2,2), dtype='complex128')
for ic in range(ICsPerProc):
    Psi = Ehrenfest_Trotter(Dt, Psi0,(x[ic],p[ic],s0))
    for ts in range(TimePoints_Dt):
        Rho_OnProc[ts] +=  GetDMatFromPsi(Psi[:,ts])   
Rho_OnProc /= ICsPerProc

SendCounts_Rho = np.array(Comm.gather(Rho_OnProc.size, 0))

if Rank != 0:
    Comm.Gatherv(sendbuf=Rho_OnProc, recvbuf=None, root=0)

if Rank == 0:
    Rho_FullList = np.empty(sum(SendCounts_Rho), dtype=Rho_OnProc.dtype)
    Comm.Gatherv(sendbuf=Rho_OnProc, recvbuf=(Rho_FullList, SendCounts_Rho), root=0)
    Rho_FullList = Rho_FullList.reshape((MC_Blocks, ProcsPerBlock, TimePoints_Dt, 2, 2))
    Rho_BlockList = np.average(Rho_FullList, axis = 1)

    P00_BlockList = np.real(Rho_BlockList.transpose((2,3,0,1))[0][0])
    P11_BlockList = np.real(Rho_BlockList.transpose((2,3,0,1))[1][1])

    P00_Avg = np.average(P00_BlockList, axis = 0)
    P11_Avg = np.average(P11_BlockList, axis = 0)

    P00_Std = np.std(P00_BlockList, axis = 0, ddof=1) / np.sqrt(MC_Blocks)
    P11_Std = np.std(P11_BlockList, axis = 0, ddof=1) / np.sqrt(MC_Blocks)

    OutFileName = f"{OutFile_BaseName}_Trotter_Ehrenfest.csv"
    ErrFileName = f"{OutFile_BaseName}_Trotter_Ehrenfest-Err.csv"

    count = 0
    with open(f"{OutFileName}",'w') as f:
        for t in Time_Dt:
            RhoString = f"{t:.10f}, {P00_Avg[count]:.10f}, {P11_Avg[count]:.10f}"
            print(f"{RhoString}",file=f)
            count += 1

    count = 0
    with open(f"{ErrFileName}",'w') as f:
        for t in Time_Dt:
            RhoString = f"{t:.10f}, {P00_Std[count]:.10f}, {P11_Std[count]:.10f}"
            print(f"{RhoString}",file=f)
            count += 1


    T_End = process_time() 
    print(f"Finished Trotter: Time = {T_End - T_Start}",flush=True)
    print(f"Time/IC = {(T_End - T_Start) / ICsPerProc}",flush=True) 

    del Rho_FullList,Rho_BlockList,P00_BlockList,P11_BlockList
    del P00_Avg,P11_Avg,P00_Std,P11_Std

del Psi,Rho_OnProc,SendCounts_Rho 

gc.collect()

# %%

###################################################################################################
#########################################  GradDescent  ###########################################                        
###################################################################################################

T_Start = process_time()

StepsTaken_OnProc = np.zeros(TimePoints_Dt, dtype='int64')
Rho_OnProc = np.zeros((TimePoints_Dt,2,2), dtype='complex128')

for ic in range(ICsPerProc):
    (StepsTaken,Psi) = Ehrenfest_VQD(None, Dt, Psi0,(x[ic],p[ic],s0))
    StepsTaken_OnProc += StepsTaken
    for ts in range(TimePoints_Dt):
        Rho_OnProc[ts] +=  GetDMatFromPsi(Psi[:,ts]) 

Rho_OnProc /= ICsPerProc

SendCounts_Rho = np.array(Comm.gather(Rho_OnProc.size, 0))
SendCounts_StepsTaken = np.array(Comm.gather(StepsTaken_OnProc.size, 0))

if Rank != 0:
    Comm.Gatherv(sendbuf=Rho_OnProc, recvbuf=None, root=0)
    Comm.Gatherv(sendbuf=StepsTaken_OnProc, recvbuf=None, root=0)

if Rank == 0:
    Rho_FullList = np.empty(sum(SendCounts_Rho), dtype=Rho_OnProc.dtype)
    Comm.Gatherv(sendbuf=Rho_OnProc, recvbuf=(Rho_FullList, SendCounts_Rho), root=0)
    Rho_FullList = Rho_FullList.reshape((MC_Blocks, ProcsPerBlock, TimePoints_Dt, 2, 2))
    Rho_BlockList = np.average(Rho_FullList, axis = 1)

    StepsTaken_FullList = np.empty(sum(SendCounts_StepsTaken), dtype=StepsTaken_OnProc.dtype)
    Comm.Gatherv(sendbuf=StepsTaken_OnProc, recvbuf=(StepsTaken_FullList, SendCounts_StepsTaken), root=0)
    StepsTaken_FullList = StepsTaken_FullList.reshape((MC_Blocks, ProcsPerBlock, TimePoints_Dt))
    StepsTaken_Avg = np.average(StepsTaken_FullList, axis = (0,1))
    StepsTaken_Avg /= ICsPerProc

    P00_BlockList = np.real(Rho_BlockList.transpose((2,3,0,1))[0][0])
    P11_BlockList = np.real(Rho_BlockList.transpose((2,3,0,1))[1][1])

    P00_Avg = np.average(P00_BlockList, axis = 0)
    P11_Avg = np.average(P11_BlockList, axis = 0)

    P00_Std = np.std(P00_BlockList, axis = 0, ddof=1) / np.sqrt(MC_Blocks)
    P11_Std = np.std(P11_BlockList, axis = 0, ddof=1) / np.sqrt(MC_Blocks)

    OutFileName   = f"{OutFile_BaseName}_GradDescent_Ehrenfest.csv"
    ErrFileName   = f"{OutFile_BaseName}_GradDescent_Ehrenfest-Err.csv"
    StepsFileName = f"{OutFile_BaseName}_GradDescent_StepsTaken.csv"

    count = 0
    with open(f"{OutFileName}",'w') as f:
        for t in Time_Dt:
            RhoString = f"{t:.10f}, {P00_Avg[count]:.10f}, {P11_Avg[count]:.10f}"
            print(f"{RhoString}",file=f)
            count += 1

    count = 0
    with open(f"{ErrFileName}",'w') as f:
        for t in Time_Dt:
            RhoString = f"{t:.10f}, {P00_Std[count]:.10f}, {P11_Std[count]:.10f}"
            print(f"{RhoString}",file=f)
            count += 1

    count = 0
    with open(f"{StepsFileName}",'w') as f:
        for t in Time_Dt:
            OutString = f"{t:.10f}, {StepsTaken_Avg[count]:.10f}"
            print(f"{OutString}",file=f)
            count += 1

    T_End = process_time() 
    print(f"Finished GradDescent: Time = {T_End - T_Start}",flush=True)
    print(f"Time/IC = {(T_End - T_Start) / ICsPerProc}",flush=True) 

    del Rho_FullList,Rho_BlockList,P00_BlockList,P11_BlockList
    del P00_Avg,P11_Avg,P00_Std,P11_Std,StepsTaken_Avg

del StepsTaken,StepsTaken_OnProc,SendCounts_StepsTaken
del Psi,Rho_OnProc,SendCounts_Rho

gc.collect()

if NoQuantum:
    sys.exit(0)

# %%

###################################################################################################
###########################################  Quantum  #############################################                        
###################################################################################################

T_Start = process_time()

StepsTaken_OnProc = np.zeros(TimePoints_Dt, dtype='int64')
Rho_OnProc = np.zeros((TimePoints_Dt,2,2), dtype='complex128')

for ic in range(ICsPerProc):
    (StepsTaken,Psi) = Ehrenfest_VQD(backend, Dt, Psi0,(x[ic],p[ic],s0))
    StepsTaken_OnProc += StepsTaken
    for ts in range(TimePoints_Dt):
        Rho_OnProc[ts] +=  GetDMatFromPsi(Psi[:,ts])         
    if Rank == 0:
        print(f"Finished IC {ic + 1}",flush=True)
Rho_OnProc /= ICsPerProc

SendCounts_Rho = np.array(Comm.gather(Rho_OnProc.size, 0))
SendCounts_StepsTaken = np.array(Comm.gather(StepsTaken_OnProc.size, 0))

if Rank != 0:
    Comm.Gatherv(sendbuf=Rho_OnProc, recvbuf=None, root=0)
    Comm.Gatherv(sendbuf=StepsTaken_OnProc, recvbuf=None, root=0)

if Rank == 0:
    Rho_FullList = np.empty(sum(SendCounts_Rho), dtype=Rho_OnProc.dtype)
    Comm.Gatherv(sendbuf=Rho_OnProc, recvbuf=(Rho_FullList, SendCounts_Rho), root=0)
    Rho_FullList = Rho_FullList.reshape((MC_Blocks, ProcsPerBlock, TimePoints_Dt, 2, 2))
    Rho_BlockList = np.average(Rho_FullList, axis = 1)

    StepsTaken_FullList = np.empty(sum(SendCounts_StepsTaken), dtype=StepsTaken_OnProc.dtype)
    Comm.Gatherv(sendbuf=StepsTaken_OnProc, recvbuf=(StepsTaken_FullList, SendCounts_StepsTaken), root=0)
    StepsTaken_FullList = StepsTaken_FullList.reshape((MC_Blocks, ProcsPerBlock, TimePoints_Dt))
    StepsTaken_Avg = np.average(StepsTaken_FullList, axis = (0,1))
    StepsTaken_Avg /= ICsPerProc

    P00_BlockList = np.real(Rho_BlockList.transpose((2,3,0,1))[0][0])
    P11_BlockList = np.real(Rho_BlockList.transpose((2,3,0,1))[1][1])

    P00_Avg = np.average(P00_BlockList, axis = 0)
    P11_Avg = np.average(P11_BlockList, axis = 0)

    P00_Std = np.std(P00_BlockList, axis = 0, ddof=1) / np.sqrt(MC_Blocks)
    P11_Std = np.std(P11_BlockList, axis = 0, ddof=1) / np.sqrt(MC_Blocks)

    OutFileName   = f"{OutFile_BaseName}_Quantum_Ehrenfest.csv"
    ErrFileName   = f"{OutFile_BaseName}_Quantum_Ehrenfest-Err.csv"
    StepsFileName = f"{OutFile_BaseName}_Quantum_StepsTaken.csv"

    count = 0
    with open(f"{OutFileName}",'w') as f:
        for t in Time_Dt:
            RhoString = f"{t:.10f}, {P00_Avg[count]:.10f}, {P11_Avg[count]:.10f}"
            print(f"{RhoString}",file=f)
            count += 1

    count = 0
    with open(f"{ErrFileName}",'w') as f:
        for t in Time_Dt:
            RhoString = f"{t:.10f}, {P00_Std[count]:.10f}, {P11_Std[count]:.10f}"
            print(f"{RhoString}",file=f)
            count += 1

    count = 0
    with open(f"{StepsFileName}",'w') as f:
        for t in Time_Dt:
            OutString = f"{t:.10f}, {StepsTaken_Avg[count]:.10f}"
            print(f"{OutString}",file=f)
            count += 1

    T_End = process_time() 
    print(f"Finished Quantum: Time = {T_End - T_Start}",flush=True)
    print(f"Time/IC = {(T_End - T_Start) / ICsPerProc}",flush=True) 

    del Rho_FullList,Rho_BlockList,P00_BlockList,P11_BlockList
    del P00_Avg,P11_Avg,P00_Std,P11_Std,StepsTaken_Avg

del StepsTaken,StepsTaken_OnProc,SendCounts_StepsTaken
del Psi,Rho_OnProc,SendCounts_Rho

gc.collect()

