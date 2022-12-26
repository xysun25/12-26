# -*- coding:utf-8 -*-
# 作者 ：路飞太郎
# 时间 : 2022/11/08
# 简述: 分析提取轨迹（abstracted_traj文件）中阴离子和阳离子的角度，即取向分布；

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis import transformations
import os,re

# 原子的相对质量分数
atomic_wt = {'H': 1.008, 'Li': 6.941, 'B': 10.811, 'C': 12.011,
             'N': 14.007, 'O': 15.9994, 'F': 18.998, 'Ne': 20.180,
             'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086,
             'P': 30.974, 'S': 32, 'Cl': 35.453, 'Ar': 39.948,
             'K': 39.098, 'Ca': 40.078, 'Ti': 47.867, 'Fe': 55.845,
             'Zn': 65.38, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
             'Mo': 95.96, 'Ru': 101.07, 'Sn': 118.710, 'Te': 127.60,
             'I': 126.904, 'Xe': 131.293}

# Takes in a file path to open, what string you start splitting at, and the indexing for each line
def fetchList(filePath, splitString, indexingStart, indexingEnd='None', skipBlankSplit=False, *args):
    readMode = False
    openedFile = open(filePath, 'r')
    listArray = []
    currentLine = ''
    for line in openedFile:
        previousLine = currentLine
        currentLine = str(line)
        if currentLine.isspace() and readMode is True:  # In each line, check if it's purely white space
            # If the previous line doesn't countain the spliting string (i.e. this isn't the line directly after the split), stop reading; only occurs if the skipBlankSplit option is set to True
            if splitString not in previousLine or skipBlankSplit is False:
                readMode = False
        if currentLine[0] == '[':  # The same is true if it's the start of a new section, signalled by '['
            readMode = False
        if readMode is True:
            if line[0] != ';' and currentLine.isspace() is False:  # If the line isn't a comment
                if indexingEnd == 'None':
                    # Just split with a start point, and take the rest
                    importantLine = (currentLine.split())[indexingStart:]
                else:
                    # Define the bigt we care about as the area between index start and end
                    importantLine = (currentLine.split())[indexingStart:indexingEnd]
                # If any additional arguments were given (and they're numbers), also add this specific index in
                for ar in args:
                    if is_number(ar):
                        importantLine += [(currentLine.split())[ar]]
                listArray.append(importantLine)  # Append the important line to the list array
        if splitString in currentLine:  # Check at the end because we only want to start splitting after the string; if found, it'll split all further lines till it gets to white space
            readMode = True
    openedFile.close()
    return listArray  # Return the array containing all associations

def read_traj(topo, traj, ilnum, waternum):
    u = mda.Universe(topo, atom_style='id resid type charge x y z')
    # add elements
    elements = []
    for i in range(len(u.atoms)):
        elements.append(chemical_symbols[np.where(types == int(u.atoms[i].type))[0][0]])
    u.add_TopologyAttr('element', values=elements)

    # add mol. name
    if waternum == 0:
        resnames = ['cation'] * ilnum + ['anion'] * ilnum
    else:
        resnames = ['cation'] * ilnum + ['anion'] * ilnum + ['water'] * waternum
    u.add_TopologyAttr('resnames', values=resnames)

    u.load_new(traj, format="LAMMPSDUMP", timeunit="fs", dt=10000)
    workflow = [transformations.unwrap(u.atoms)]
    u.trajectory.add_transformations(*workflow)
    return u

def atomic_symbol(name):
    if name[:2] in atomic_wt:
        return name[:2]
    elif name[0] in atomic_wt:
        return name[0]
    else:
        print('warning: unknown symbol for atom ' + name)
        return name

directory = './'
filename = 'data.lmp'
## elements
elements = fetchList(filename, 'Masses', 0,4, skipBlankSplit=True)
digitspattern = r'#'
types = []
chemical_symbols = []
for element in elements:
    types.append(int(element[0]))
    txt = re.sub(digitspattern, '', element[-1])
    chemical_symbols.append(atomic_symbol(txt))
types = np.array(types)
print('types:            ',types)
print('chemical_symbols: ',chemical_symbols)

il_num = 500
# full: atom-ID molecule-ID atom-type q x y z
u = mda.Universe('result.data', atom_style='id resid type charge x y z')
print(u)
# resnames
resnames = ['cati']*il_num+['anio']*il_num+['MXene']*1+['MXene']*1
u.add_TopologyAttr('resnames',values=resnames)
u.load_new("q.xyz", format="XYZ",timeunit="fs",dt=10000)
print('frames',u.trajectory.n_frames)
print("************************************************************************")
print("计算阳离子的取向分布（角度）")
print("************************************************************************")

cations = u.select_atoms('resname cati') # select all cations
resids_cations = np.unique(cations.atoms.resids) # extract resid of all cations

cation_coms = []
for ts in u.trajectory:
    for i in resids_cations:
        sel = u.select_atoms('resid %d'%i)
        com = sel.center_of_mass()
        cation_coms.append([com[0],com[1],com[2]])

cation_oriens = []
for ts in u.trajectory:
    for i in resids_cations:
        sel = u.select_atoms('resid %d'%i)
        head = u.select_atoms('resid %d and type 1'%i)
        tail = u.select_atoms('resid %d and type 3'%i)
        com = head.center_of_mass()
        vec = tail.center_of_mass()-com
        vec_norm = np.linalg.norm(vec)
        vec = vec/vec_norm  # normalized
        cation_oriens.append([vec[0],vec[1],vec[2]])
z = np.array([1,0,0])
costheta = [np.dot(z,cation_oriens[i]) for i in range(len(cation_oriens))]
costheta = np.array(costheta)

# 转变为角度
angle_d = []
for i in costheta:
    a_hu = np.arccos(i)
    a_d = a_hu*180/np.pi
    angle_d.append(a_d)
angle_d = np.array(angle_d)

# print(angle_dl.shape)
# print(costheta.shape)
cation_props = np.hstack([cation_coms,cation_oriens,costheta[:,None],angle_d[:,None]])
cation_props_pd = pd.DataFrame(cation_props,columns=['cmx', 'cmy', 'cmz','ux', 'uy', 'uz', 'costheta', 'degree'])

# print(cation_props_pd)

# 筛选出指定区域内的阳离子质心和cos,对行遍历
Y_list = np.arange(-3.5,3.5,1).tolist()
for Y in Y_list:
    x_0 = float("-25.443")
    x_1 = float("24.565")
    y_0 = float(Y)
    y_1 = float(Y + 1)
    z_0 = float("52")
    z_1 = float("96")
    anglist = []
    for index,row in cation_props_pd.iterrows():
        if (row['cmx'] >= x_0 )& (row['cmx'] <= x_1)& (row['cmy'] >= y_0 )& (row['cmy'] <= y_1)& (row['cmz'] >= z_0 )& (row['cmz']  <= z_1):
            anglist.append(row['degree'])
    # print(anglist)
    len_ang = len(anglist)
    ang_list = np.arange(0, 180, 10).tolist()
    for a in ang_list:
        ang_0 = float(a)
        ang_1 = float(a + 10)
        ang_00 = int(ang_0)
        ang_11 = int(ang_1)
        num = 0
        for ang in anglist:
            if ang <= ang_1 and ang >= ang_0:
                num = num + 1
        if len_ang != 0:
            probability = num / len_ang/102
            print(probability)
        if len_ang == 0:
            probability = 0
            ang_list = np.arange(0, 180, 10).tolist()
            print(probability)
    print("以上是阳离子10°角度间隔的概率分布")





