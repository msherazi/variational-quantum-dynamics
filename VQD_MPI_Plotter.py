import csv
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

with open('10k_Direct_Ehrenfest.csv', newline='') as f:
    reader = csv.reader(f)
    direct = list(reader)

with open('10k_GradDescent_Ehrenfest.csv', newline='') as g:
    reader = csv.reader(g)
    grad = list(reader)

with open('10k_Quantum_Ehrenfest.csv', newline='') as h:
    reader = csv.reader(h)
    quantum = list(reader)

list1 = []
time1 = []

for i in range(0, len(direct)):
    time1.append(direct[i][0])
    list1.append(direct[i][1])

direct = list1

list2 = []
time2 = []

for i in range(0, len(grad)):
    time2.append(grad[i][0])
    list2.append(grad[i][1])

grad = list2

list3 = []
time3 = []

for i in range(0, len(quantum)):
    time3.append(quantum[i][0])
    list3.append(quantum[i][1])

quantum = list3

directArr = np.array(direct, dtype=np.float64)
gradArr = np.array(grad, dtype=np.float64)
quantumArr = np.array(quantum, dtype=np.float64)

timeArr1 = np.array(time1, dtype=np.float64)
timeArr2 = np.array(time2, dtype=np.float64)
timeArr3 = np.array(time3, dtype=np.float64)

plt.figure(dpi=250, layout='constrained')
plt.plot(timeArr1, directArr, label=r'Direct', color = "black")
plt.plot(timeArr2, gradArr, label=r'GradDescent', color = "red")
plt.plot(timeArr3, quantumArr, label=r'Quantum', color = "orange")

plt.title(r"Time-Evolution of Wave Function")
plt.xlabel(f'Time')
plt.ylabel(r'$\Psi(t) = [\psi(t) \quad \phi(t)]^\mathrm{T}$')
plt.legend(ncol=2)
plt.savefig("plot5", dpi=300)
plt.show()
