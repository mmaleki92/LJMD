import matplotlib.pyplot as plt
import numpy as np

x = np.array([5.757963,
2.938518,
1.942482,
1.415247,
1.230259,
0.953432,
0.92385,
0.890861,
0.68203,
0.957637,
0.95797,
0.498783])/10000
nps = [1, 11, 16, 32]

plt.plot(range(1,13), x)
cell_list = np.array([3.039])/10000
np_serial = [1]
plt.scatter(np_serial, cell_list)
plt.title("$Argon$ 108 MPI")
plt.xlabel("np")
plt.ylabel("Time(sec)")
plt.show()

x_argn_2916 = np.array([222.69735, 23.598347, 17.532334, 10.096639])/1000
nps = [1, 11, 16, 32]

cell_list = np.array([23.66])/1000
np_serial = [1]
plt.scatter(np_serial, cell_list)

plt.plot(nps, x_argn_2916)
plt.title("$Argon$ 2916 MPI")
plt.xlabel("np")
plt.ylabel("Time(sec)")
plt.show()

# x_argn_78732 = np.array([208, 82.551869])/20 
# nps = [2, 32]
# cell_list = np.array([23.66])/10000
# np_serial = [1]
# plt.scatter(np_serial, cell_list)

# plt.plot(nps, x_argn_78732)
# plt.title("$Argon$ 2916 MPI")
# plt.xlabel("np")
# plt.ylabel("Time(sec)")
# plt.show()

#############

x_argn_78732 = np.array([208/5, 82.551869/20])
nps = [2, 32]

cell_list = np.array([14.423])/20
np_serial = [1]
plt.scatter(np_serial, cell_list)
plt.plot(nps, x_argn_78732)
plt.title("$Argon$ 78732 MPI")
plt.xlabel("np")
plt.ylabel("Time(sec)")
plt.show()







import matplotlib.pyplot as plt
import numpy as np

x = np.array([5.757963,
2.938518,
1.942482,
1.415247,
1.230259,
0.953432,
0.92385,
0.890861,
0.68203,
0.957637,
0.95797,
0.498783])/10000
nps = [1, 11, 16, 32]

plt.plot(range(1,13), x)
cell_list = np.array([3.039])/10000
np_serial = [1]

x_argn_2916 = np.array([222.69735, 23.598347, 17.532334, 10.096639])/1000
nps = [1, 11, 16, 32]

cell_list = np.array([23.66])/1000
np_serial = [1]
# plt.scatter(np_serial, cell_list)

plt.plot(nps, x_argn_2916)

x_argn_78732 = np.array([208/5, 82.551869/20])
nps = [2, 32]

cell_list = np.array([14.423])/20
np_serial = [1]
plt.scatter(np_serial, cell_list)
plt.plot(nps, x_argn_78732)
plt.title("$Argon$ 78732 MPI")
plt.xlabel("np")
plt.ylabel("Time(sec)")
plt.show()




