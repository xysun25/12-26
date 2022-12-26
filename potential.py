import sys
import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


class info_mol():
    def __init__(self) -> None:
        pass

    def read_info_mol(self,file):
        atom_in_mol = []
        charge_of_atom = []
        mass_of_atom = []
        if os.path.exists(file):
            f1 = open(file,'r')
            f1.readline()
            atom_in_mol = int(f1.readline())
            print("the atom number in the molecule is : %d"%atom_in_mol)
            for i in range(atom_in_mol):
                line = f1.readline().split()
                charge_of_atom.append(line[5])
            f1.close()
        else:
            raise IOError('the %s is not existed'%files)
        return atom_in_mol,charge_of_atom

cation_inf_instant = info_mol()
anion_inf_instant = info_mol()
atom_num_in_cation,charge_of_cation=cation_inf_instant.read_info_mol("EmimOH.mmol")
atom_num_in_anion,charge_of_anion=anion_inf_instant.read_info_mol("Tf2n.mmol")
charge_of_cation = list(map(float,charge_of_cation))
charge_of_anion = list(map(float,charge_of_anion))

# charge_list = charge_of_cation*500+charge_of_anion*500
# charge_list = charge_of_cation*500
charge_list = charge_of_anion*500



# dataframe of charge of 17500 atoms
char = {"charge":charge_list}
df1 = DataFrame(char)

# dataframe of the last frame of trajectory getting from abstract_trj.py
file = "abstracted_traj"
data1 = pd.read_table(r"abstracted_traj",sep = "  ",header=1)
data2 = data1.iloc[0:264200,]
new_name = ['Atom','x','y','z']
data3 = data2['Atoms. Timestep: 3000000'].str.split(' ',expand = True)
data3.columns = new_name

# dataframe of 'Atom','x','y','z' and 'charge'
data4 = pd.concat([data3, df1], axis=1)
data4[['x','y','z']] = data4[['x','y','z']].apply(pd.to_numeric)

# print(data4)

z_list = np.arange(0,50,2.5).tolist()
for i in z_list:
    list = []
    x_0=float("-25.443")
    x_1=float("24.565")
    y_0=float("-14")
    y_1=float("14")
    z_0=float(i)
    z_1=float(i+2.5)
    volume=(x_1-x_0)*(y_1-y_0)*(z_1-z_0)
    data5 =data4[(data4['x'] >= x_0 )& (data4['x'] <= x_1)& (data4['y'] >= y_0 )& (data4['y'] <= y_1)& (data4['z'] >= z_0 )& (data4['z'] <= z_1)]
    number_of_charge = data5['charge'].sum()
    density_of_charge = number_of_charge/volume
    # print(density_of_charge)
    # fai = ((z_0+z_1)/2+50)*(-4)*np.pi*density_of_charge
    fai = (50-((z_0+z_1)/2))*(-4)*np.pi*density_of_charge
    print(fai)


# # calculate the density of charge
# list=[]
# list=input("please input 6 numbers for the range of x y z 3directions:").split(',')
# x_0=float(list[0])
# x_1=float(list[1])
# y_0=float(list[2])
# y_1=float(list[3])
# z_0=float(list[4])
# z_1=float(list[5])
# volume=(x_1-x_0)*(y_1-y_0)*(z_1-z_0)
#
# data5 =data4[(data4['x'] >= x_0 )& (data4['x'] <= x_1)& (data4['y'] >= y_0 )& (data4['y'] <= y_1)& (data4['z'] >= z_0 )& (data4['z'] <= z_1)]
#
# number_of_charge = data5['charge'].sum()
# density_of_charge = number_of_charge/volume
# print(density_of_charge)
#
