from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdDistGeom
import pandas as pd
import os

# 读取目标文件
df = pd.read_csv('data.csv')
df.head(5)
len(df)

i = 1
image_list = list()
for smi in df['Canonical SMILES']:
    # 让生成的分子图更好看（该图片与薛定谔渲染相似）
    rdDepictor.SetPreferCoordGen(True)
    # 如果成功生成分子则将分子名称存至image列表，如果smiles有错误，则添加None
    try:
        # 将smiles转为mol格式
        mol = Chem.MolFromSmiles(smi)
        # 绘制分子
        img = Draw.MolToImage(mol)
        image = "mol%d.png"%(i)
        #存为图片 并添加至列表
        img.save(image)
        image_list.append(image)
    except:
        image_list.append(None)
        print(f"in line {i} smiles: {smi} has error")
    i += 1

# 将存有分子图片文件名的的分子列表转为数据帧 并与原始数据集合并
df_image = pd.DataFrame(image_list,columns=['image'])
# 合并
df_new = pd.concat([df,df_image],axis=1)
# 检查前五个是否有错误
df_new.head(5)

# 生成SDF文件
i = 1
sdf_list = list()
for smi in df['Canonical SMILES']:
    try:
        mol = Chem.MolFromSmiles(smi)
        # 加氢
        mol = Chem.AddHs(mol)
        # 生成3d分子结构
        etkdgv3 = rdDistGeom.ETKDGv3()
        rdDistGeom.EmbedMolecule(mol,etkdgv3)
        # mmff94力场优化分子构象
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        # 存为sdf文件 并为列表添加文件名
        filename = f'molecule{i}.sdf'
        writer = Chem.SDWriter(filename)
        writer.write(mol)
        writer.close()
        sdf_list.append(filename)
    except:
        sdf_list.append(None)
        print(f"in line {i} smiles: {smi} has error")
    i += 1

df_sdf = pd.DataFrame(sdf_list,columns=['sdf'])
df_image_sdf = pd.concat([df_new,df_sdf],axis=1)
df_image_sdf.head(5)

# 生成PDB文件
i = 1
pdb_list = list()
for smi in df['Canonical SMILES']:
    try:
        mol = Chem.MolFromSmiles(smi)
        # 加氢
        mol = Chem.AddHs(mol)
        # 生成3d分子结构
        etkdgv3 = rdDistGeom.ETKDGv3()
        rdDistGeom.EmbedMolecule(mol,etkdgv3)
        # mmff94力场优化分子构象
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        # 存为pdb文件 并为列表添加文件名
        filename = f'molecule{i}.pdb'
        writer = Chem.PDBWriter(filename)
        writer.write(mol)
        writer.close()
        pdb_list.append(filename)
    except:
        pdb_list.append(None)
        print(f"in line {i} smiles: {smi} has error")
    i += 1

df_pdb = pd.DataFrame(pdb_list,columns=['pdb'])
df_image_sdf_pdb = pd.concat([df_image_sdf,df_pdb],axis=1)
df_image_sdf_pdb.head(5)

# 生成MOL文件
i = 1
mol_list = list()
for smi in df['Canonical SMILES']:
    try:
        mol = Chem.MolFromSmiles(smi)
        # 加氢
        mol = Chem.AddHs(mol)
        # 生成3d分子结构
        etkdgv3 = rdDistGeom.ETKDGv3()
        rdDistGeom.EmbedMolecule(mol,etkdgv3)
        # mmff94力场优化分子构象
        AllChem.MMFFOptimizeMoleculeConfs(mol)
        # 存为mol文件 并为列表添加文件名
        filename = f'molecule{i}.mol'
        Chem.MolToMolFile(mol,filename)
        mol_list.append(filename)
    except:
        mol_list.append(None)
        print(f"in line {i} smiles: {smi} has error")
    i += 1

df_mol = pd.DataFrame(mol_list,columns=['mol'])
df_image_sdf_pdb_mol = pd.concat([df_image_sdf_pdb,df_mol],axis=1)
df_image_sdf_pdb_mol.head(5)

## 将结果保存
df_image_sdf_pdb_mol.to_csv('file_format.csv',index=None)