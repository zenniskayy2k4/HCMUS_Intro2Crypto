import numpy as np

def string_to_matrix(s, n):
    numbers = [ord(char) - ord('A') for char in s.upper()]
    while len(numbers) % n != 0:
        numbers.append(ord('X') - ord('A'))
    matrix = [numbers[i:i + n] for i in range(0, len(numbers), n)]
    return matrix

def matrix_to_string(matrix):
    chars = [chr(num + ord('A')) for row in matrix for num in row]
    return ''.join(chars)

# Define matrices
B = np.array([[5],
              [6]])
K = np.array([[2, 3],
              [1, 4]])
T = np.array([[0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0]])

# s = "FACULTYOFIT"
s = "FACULTYOFINFORMATIONTECHNOLOGY"
# s = "SCOOBYDOOWHEREAREYOU"
n = B.shape[0]

# Convert string to matrix A
A = np.array(string_to_matrix(s, n)).T

# Step 1: First substitution with K, B
C = np.hstack([(np.dot(K, A[:, i].reshape(-1, 1)) + B) % 26 for i in range(A.shape[1])])
print("Matrix C:")
print(C)

# Step 2: Permutation with T
D = []
C_vec = C.flatten('F')  # Convert to column-major vector
T_len = T.shape[0]

# Pad with value 24 (X) if needed
if len(C_vec) % T_len != 0:
    padding_length = T_len - (len(C_vec) % T_len)
    C_vec = np.pad(C_vec, (0, padding_length), constant_values=24)

# Process in groups of T_len
for i in range(0, len(C_vec), T_len):
    group = C_vec[i:i+T_len].reshape(-1, 1)
    permuted_group = np.dot(T, group) % 26
    D.append(permuted_group.flatten())

D = np.array(D).T
print("\nMatrix D before reshape:")
print(D)

# Pad D if needed and reshape into n×(size/n) matrix
D_flat = D.flatten('F')  # Flatten D in column-major order

# Pad with value 24 (X) if needed
if len(D_flat) % n != 0:
    padding_length = n - (len(D_flat) % n)
    D_flat = np.pad(D_flat, (0, padding_length), constant_values=24)

# Reshape into n×(size/n) matrix
D = D_flat.reshape(n, -1, order='F')
print("\nMatrix D after reshape:")
print(D)

# Step 3: Second substitution with K, B
E = np.hstack([(np.dot(K, D[:, i].reshape(-1, 1)) + B) % 26 for i in range(D.shape[1])])
print("\nMatrix E:")
print(E)

result = matrix_to_string(E.T)
print("\nEncrypted result:")
print(result)