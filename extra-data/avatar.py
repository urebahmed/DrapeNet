import numpy as np

def load_obj(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

def compute_transformation_matrices(vertices, num_matrices=24):
    # Placeholder for actual transformation logic
    tfs_c = np.array([np.eye(4, dtype=np.float32) for _ in range(num_matrices)])
    tfs_c_inv = np.array([np.linalg.inv(tf) for tf in tfs_c])
    return tfs_c, tfs_c_inv

# Load the OBJ file
obj_file_path = '/home/hello/drapenet/result.obj'
vertices = load_obj(obj_file_path)
print(f"Loaded {vertices.shape[0]} vertices.")

# Compute transformation matrices
tfs_c, tfs_c_inv = compute_transformation_matrices(vertices)

# Save the NPZ file
output_npz_path = '/home/hello/drapenet/DrapeNet/extra-data/avatar_info.npz'
np.savez(output_npz_path, tfs_c=tfs_c, tfs_c_inv=tfs_c_inv)
print(f"Saved transformation matrices to {output_npz_path}")
