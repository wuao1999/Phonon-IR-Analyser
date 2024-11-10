import numpy as np
import pandas as pd
import spglib
from ase.io import read

def get_symmetry_from_poscar(poscar_file, threshold=1e-10):
    """ Take POSCAR (vasp format) as input and find symmetry operations from 
    the crystal structure and site tensors.

    Input:
        1.POSCAR file

    Output:
        1.fractional coordinates
        2.rotations&translations as paired
    """
    atoms = read(poscar_file, format='vasp')
    cell = (atoms.get_cell(),
            atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    axis = atoms.get_cell()
    frac_coordinates = atoms.get_scaled_positions()
    symmetry = spglib.get_symmetry(cell, symprec=1e-5)
    rotations = np.array([rotation.T for rotation in symmetry['rotations']])
    translations = symmetry['translations']
    cleaned_translations = translations.copy()
    cleaned_translations[abs(cleaned_translations) < threshold] = 0
    
    return axis, frac_coordinates, rotations, cleaned_translations

def get_vector_representation(rotations):
    """Get rotational parts of the symmetry operations.
    Note: Pay attention that the rotations are the tensor composition of 
    rotation operator R on a lattice basis (other than on cartesian basis),
    so the rotations are not essentially the "vector representation" that transform 
    just like (x, y, z). But they share the same traces.

    Input:
        1.rotational parts of operations.

    Output:
        1.A list of vector representations in the same order of the operations.
            Dim = (operations, 3, 3)
    """
    vec_rep = np.zeros((len(rotations), 3, 3))

    for i, rotation in enumerate(rotations):
        vec_rep[i] = rotation

    return vec_rep

def get_vector_representation_real(axis, rotations_lattice_basis, threshold=1e-10):
    """Give real vector representations
    """
    vec_rep = np.zeros((len(rotations_lattice_basis), 3, 3))

    for i, rotation in enumerate(rotations_lattice_basis):
        vec_rep[i] = np.linalg.inv(axis)@rotation@axis
        vec_rep[i] = np.where(np.abs(vec_rep[i]) < threshold, 0.0, vec_rep[i])
        vec_rep[i] = np.round(vec_rep[i], decimals=10)
    return vec_rep

def get_position_representation(rotations, translations, frac_coordinates, threshold=1e-5):
    """Get position representations, which is a mapping for atomic sites correlation under 
    certain transform.

    Input:
        1.rotational parts of operations.
        2.translational parts of operations.
        3.atomic sites

    Output:
        1.A list of position representations in the same order of the operations.
            Dim = (operations, natoms, natoms) 
    """
    n_atoms = len(frac_coordinates)
    pos_rep = np.zeros((len(rotations), n_atoms, n_atoms))

    for i in range(len(rotations)):
            transformed_pos = frac_coordinates@rotations[i]+translations[i]
            transformed_pos = transformed_pos % 1.0
            # find equivalent atom site after the transformation
            for j, coord1 in enumerate(frac_coordinates):
                 for k, coord2 in enumerate(transformed_pos):
                      diff = (coord1 - coord2) % 1.0
                      diff = np.minimum(diff, 1.0 - diff)
                      if np.all(diff < threshold):
                        pos_rep[i,j,k] = 1

    return pos_rep

def mechanical_representation(vec_rep, pos_rep):
    """Calculate the Kronecker product of vector representation and position representation
    to construct mechanical representation.

    Input:
        1.vector representations
        2.position representations

    Output:
        1.A list of mechanical representations in the same order of the operations.
            Dim = (operations, 3*natoms, 3*natoms) 
    """
    dim = vec_rep.shape[1]*pos_rep.shape[1]
    mech_rep = np.zeros((len(vec_rep), dim, dim))

    for i in range(len(vec_rep)):
        mech_rep[i] = np.kron(vec_rep[i], pos_rep[i])

    return mech_rep

def CharacterTable():
    """ Character table for D6h """
#     index = pd.MultiIndex.from_tuples([
#    ('GM1+', 'A1g'), ('GM1-', 'A1u'),
#    ('GM2+', 'A2g'), ('GM2-', 'A2u'), 
#    ('GM4+', 'B2g'), ('GM4-', 'B2u'),
#    ('GM3+', 'B1g'), ('GM3-', 'B1u'),
#    ('GM6+', 'E2g'), ('GM6-', 'E2u'),
#    ('GM5+', 'E1g'), ('GM5-', 'E1u')])
    index = ["A1g", "A1u", "A2g", "A2u", "B2g", "B2u", "B1g", "B1u", "E2g", "E2u", "E1g", "E1u"]

    return(pd.DataFrame({
    'C1': [1,1,1,1,1,1,1,1,2,2,2,2],
    'C2': [1,1,-1,-1,1,1,-1,-1,0,0,0,0],
    'C3': [1,1,1,1,-1,-1,-1,-1,2,2,-2,-2],
    'C4': [1,1,1,1,1,1,1,1,-1,-1,-1,-1],
    'C5': [1,1,-1,-1,-1,-1,1,1,0,0,0,0],
    'C6': [1,1,1,1,-1,-1,-1,-1,-1,-1,1,1],
    'C7': [1,-1,1,-1,1,-1,1,-1,2,-2,2,-2],
    'C8': [1,-1,-1,1,1,-1,-1,1,0,0,0,0],
    'C9': [1,-1,1,-1,-1,1,-1,1,2,-2,-2,2],
    'C10': [1,-1,1,-1,1,-1,1,-1,-1,1,-1,1],
    'C11': [1,-1,-1,1,-1,1,1,-1,0,0,0,0],
    'C12': [1,-1,1,-1,-1,1,-1,1,-1,1,1,-1]
    }, index=index))

class SymmetryOperation:
    # Create group operation objects
    def __init__(self, number, matrix, seitz_symbol, conjugacy_class):
        self.number = number
        self.matrix = matrix
        self.seitz_symbol = seitz_symbol
        self.conjugacy_class = conjugacy_class
        self.irreps = CharacterTable()[conjugacy_class]

def get_character_table(rotations, operations):
    """ Reordering the characters in consistent with the current order of operations 
    
    Input:
        1.rotational parts from spglib
        2.rotational parts from initialzed symmetry operation objects

    Output:
        1.character table
            Dim = (irreps, operations)
        2.symbols for irreps
    """
    IRs = operations[0].irreps.index.tolist()
    character_table = np.zeros((len(IRs), len(rotations)))

    for idx_ir, ir in enumerate(IRs):
        for i, rotation in enumerate(rotations):
            for operation in operations:
                if np.allclose(rotation, operation.matrix):
                    character_table[idx_ir][i] = operation.irreps[ir]

    return character_table, IRs

def decompose_irrep(mech_rep, character_table):
    """ Calculate the decomposition cooefficient of the irreps.

    Input:
        1.mechanical representations
        2.character table

    Output:
        1.multiplicities of irreps
            Dim = (irreps)
    """
    h = len(mech_rep) # rank of group
    n_irreps = len(character_table)
    multiplicities = np.zeros(n_irreps)

    characters = np.trace(mech_rep, axis1=1, axis2=2)

    for i in range(n_irreps):
        multiplicities[i] = (1/h) * np.sum(characters * character_table[i])
    
    return multiplicities

def diplay_decomposition(multiplicities, IRs):
    """ Display the decompostion. """
    direct_plus = []
    for i, multiplicity in enumerate(multiplicities):
        if multiplicity != 0.:
            direct_plus.append(f'{int(multiplicity)}{IRs[i]}')
    print("Γ = " + " ⊕ ".join(direct_plus))


if __name__ == "__main__":
    # 初始化点群的对称操作
    operations = [
    SymmetryOperation(1, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), "1", "C1"),
    SymmetryOperation(2, np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]]), "3+_001", "C4"),
    SymmetryOperation(3, np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]), "3-_001", "C4"),
    SymmetryOperation(4, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]), "2_001", "C3"),
    SymmetryOperation(5, np.array([[0, 1, 0], [-1, 1, 0], [0, 0, 1]]), "6-_001", "C6"),
    SymmetryOperation(6, np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]]), "6+_001", "C6"),
    SymmetryOperation(7, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]), "2_110", "C2"),
    SymmetryOperation(8, np.array([[1, -1, 0], [0, -1, 0], [0, 0, -1]]), "2_100", "C2"),
    SymmetryOperation(9, np.array([[-1, 0, 0], [-1, 1, 0], [0, 0, -1]]), "2_010", "C2"),
    SymmetryOperation(10, np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]), "2_1-10", "C5"),
    SymmetryOperation(11, np.array([[-1, 1, 0], [0, 1, 0], [0, 0, -1]]), "2_120", "C5"),
    SymmetryOperation(12, np.array([[1, 0, 0], [1, -1, 0], [0, 0, -1]]), "2_210", "C5"),
    SymmetryOperation(13, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]), "-1", "C7"),
    SymmetryOperation(14, np.array([[0, 1, 0], [-1, 1, 0], [0, 0, -1]]), "-3_+001", "C10"),
    SymmetryOperation(15, np.array([[1, -1, 0], [1, 0, 0], [0, 0, -1]]), "-3_-001", "C10"),
    SymmetryOperation(16, np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]), "m_001", "C9"),
    SymmetryOperation(17, np.array([[0, -1, 0], [1, -1, 0], [0, 0, -1]]), "-6_-001", "C12"),
    SymmetryOperation(18, np.array([[-1, 1, 0], [-1, 0, 0], [0, 0, -1]]), "-6_+001", "C12"),
    SymmetryOperation(19, np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]), "m_110", "C8"),
    SymmetryOperation(20, np.array([[-1, 1, 0], [0, 1, 0], [0, 0, 1]]), "m_100", "C8"),
    SymmetryOperation(21, np.array([[1, 0, 0], [1, -1, 0], [0, 0, 1]]), "m_010", "C8"),
    SymmetryOperation(22, np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), "m_1-10", "C11"),
    SymmetryOperation(23, np.array([[1, -1, 0], [0, -1, 0], [0, 0, 1]]), "m_120", "C11"),
    SymmetryOperation(24, np.array([[-1, 0, 0], [-1, 1, 0], [0, 0, 1]]), "m_210", "C11")
    ]
    # 不可约表示：
    # IRs = ["A1g", "A1u", "A2g", "A2u", "B2g", "B2u", "B1g", "B1u", "E2g", "E2u", "E1g", "E1u"]
    # poscar_file = "POSCAR_graphene"
    poscar_file = "POSCAR_MnTe"
    # 读取POSCAR
    axis, frac_coordinates, rotations, translations = get_symmetry_from_poscar(poscar_file)
    # 获取矢量表示
    vec_rep = get_vector_representation(rotations)
    # 获取位置表示
    pos_rep = get_position_representation(rotations, translations, frac_coordinates)
    # 直积得到力学表示
    mech_rep = mechanical_representation(vec_rep, pos_rep)
    # 检索适配的特征标表
    character_table, IRs = get_character_table(rotations, operations)
    # 计算得到不可约表示对应的重数
    multiplicities = decompose_irrep(mech_rep, character_table)
    # 显示分解结果
    print(f"Run factor group analysis for {poscar_file.split('_')[-1]}")
    diplay_decomposition(multiplicities, IRs)
