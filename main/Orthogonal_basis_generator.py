from IRA_gam import *

def numpy_to_mathematica(matrix):
    """
    将numpy复数矩阵转换为Mathematica矩阵格式的字符串
    
    Parameters:
    matrix: numpy.ndarray - 复数numpy矩阵
    
    Returns:
    str - Mathematica格式的矩阵字符串
    """
    def format_complex(num):
        # 处理很小的数字（接近0）
        real = np.real(num)
        imag = np.imag(num)
        
        # 设置阈值，小于此值视为0
        threshold = 1e-10
        
        if abs(real) < threshold: real = 0
        if abs(imag) < threshold: imag = 0
        
        # Mathematica复数格式化
        if real == 0 and imag == 0:
            return "0"
        elif imag == 0:
            return f"{real}"
        elif real == 0:
            return f"{imag} I"
        else:
            return f"{real} + {imag} I"

    # 构建Mathematica矩阵字符串
    rows = []
    for row in matrix:
        formatted_row = [format_complex(num) for num in row]
        rows.append("{" + ", ".join(formatted_row) + "}")
    
    mathematica_matrix = "{\n" + ",\n".join(rows) + "\n}"
    return mathematica_matrix

def methematica_matrix2txt(mathematica_matrix):
    with open('mathematica_matrix.txt', 'w') as f:
        f.write(f"matrix = {mathematica_matrix}\n")


if __name__ == "__main__":
    # 初始化点群的对称操作及特征标
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
    # if __name__ == "__main__":
    # poscar_file = "POSCAR_graphene"
    poscar_file = "POSCAR_MnTe"
    # 读取POSCAR
    cell, frac_coordinates, rotations, translations = get_symmetry_from_poscar(poscar_file)
    # 获取矢量表示
    vec_rep = get_vector_representation_real(cell, rotations)
    # 获取位置表示
    pos_rep = get_position_representation(rotations, translations, frac_coordinates)
    # 获取力学表示
    mech_rep = mechanical_representation(pos_rep, vec_rep)
    # 检索适配的特征标表
    character_table, IRs = get_character_table(rotations, operations)

    # 这里需要调取不可约表示的数据
    """ https://github.com/spglib/spgrep """
    from spgrep import get_spacegroup_irreps

    irreps, rots, translations, mapping_little_group = get_spacegroup_irreps(cell, frac_coordinates, numbers=[1, 1, 1, 1], kpoint=[0, 0, 0])
    projection_operator11 = np.zeros(mech_rep.shape, dtype='complex')
    projection_operator12 = np.zeros(mech_rep.shape, dtype='complex')
    projection_operator22 = np.zeros(mech_rep.shape, dtype='complex')
    projection_operator = np.zeros(mech_rep.shape, dtype='complex')

    # IRs = ["A1g", "A1u", "A2g", "A2u", "B2g", "B2u", "B1g", "B1u", "E2g", "E2u", "E1g", "E1u"]
    # Γ = 2A2u ⊕ 1B2u ⊕ 1B1g ⊕ 1E2g ⊕ 1E2u ⊕ 2E1u
    # 3 5 6 8 9 11

    irreps_index = 8
    for irrep in irreps:
        if np.allclose(np.trace(irrep, axis1=1, axis2=2),character_table[irreps_index]):
            if irrep.shape[-2:] == (1,1):
                for i, rotation in enumerate(rotations):
                    projection_operator11[i] = mech_rep[i]*irrep[i][0,0]
                O_trace = (1/24)*np.sum(projection_operator11, axis=0)
            if irrep.shape[-2:] == (2,2):
                for i, rotation in enumerate(rotations):
                    projection_operator11[i] = mech_rep[i]*irrep[i][0,0]
                    projection_operator12[i] = mech_rep[i]*irrep[i][0,1]
                    projection_operator22[i] = mech_rep[i]*irrep[i][1,1]
                    projection_operator[i] = mech_rep[i]*(irrep[i][0,0]+irrep[i][1,1])
                O11 = (1/24)*np.sum(projection_operator11, axis=0)
                O12 = (1/24)*np.sum(projection_operator12, axis=0)
                O22 = (1/24)*np.sum(projection_operator22, axis=0)
                O_trace = (1/24)*np.sum(projection_operator, axis=0)

    # mathematica_matrix11 = numpy_to_mathematica(O11)
    # mathematica_matrix12 = numpy_to_mathematica(O12)
    mathematica_matrix_trace = numpy_to_mathematica(O_trace)
    methematica_matrix2txt(mathematica_matrix_trace)