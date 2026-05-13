def dot(x: list, y: list) -> float: 
    """
    Calculate dot product of two vectors:   

        x * y = Σ x[i] * y[i]

    Args:
        x: first vector
        y: second vector
    Returns:
        float: dot product of two vectors
    """

    if len(x) != len(y):
        raise ValueError("Vector must have same length")
    
    s = 0.0
    for i in range(len(x)):
        s += x[i] * y[i]
    return s

def vector_add(x: list[float], y: list[float]) -> list[float]:
    """
    Calculate the sum of two vectors:

        x + y = [x[0] + y[0], ..., x[n] + y[n]]
    
    Args:
        x: first vector
        y: second vector
    Returns:
        list: sum of two vectors
    """

    if len(x) != len(y):
        raise ValueError("Vector must have same length")
    
    result = []
    for i in range(len(x)):
        result.append(x[i] + y[i])
    return result

def vector_sub(x: list[float], y: list[float]) -> list[float]: 
    """
    Subtract two vectors: 

        x - y = [x[0] - y[0], ..., x[n] - y[n]]
    
    Args:
        x: first vector
        y: second vector
    Returns:
        list: difference of two vectors
    """
    if len(x) != len(y):
        raise ValueError("Vector must have same length")
    
    result = []
    for i in range(len(x)):
        result.append(x[i] - y[i])
    return result

def scalar_multiply(x: list[float], k: float) -> list[float]:
    """
    Multiply a vector by a scalar:

        k * x = [k*x[0], ..., k*x[n]]

    Args:
        x: input vector
        k: scalar value

    Returns:
        list: scaled vector
    """

    result = []

    for i in range(len(x)):
        result.append(k * x[i])

    return result

def norm(v: list[float]) -> float: 
    """
    Calculate Euclidean norm of a vector:

        ||v|| = sqrt(v[0]^2 + ... + v[n]^2)
    
    Args:
        v: input vector
    Returns:
        float: Euclidean norm of the vector
    """
    
    s = 0.0
    for val in v:
        s += val * val
    
    return s**0.5

def normalize(v: list) -> list[float]:
    """
    Normalize a vector:

        v / ||v||
    
    Args:
        v: input vector
    Returns:
        list: normalized vector
    """
    return scalar_multiply(1/norm(v), v)

def transpose(A: list[list[float]]) -> list[list[float]]:
    """
    Calculate transposed matrix:

        A^T = [a_{ji}]
    
    Args:
        A: input matrix
    Returns:
        list: transpose matrix
    """
    m = len(A)
    n = len(A[0])

    AT = [[0.0 for _ in range(m)] for _ in range(n)]

    for i in range(m):
        for j in range(n): 
            AT[j][i] = A[i][j]
            
    return AT

def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    Calculate matrix multiplication: 
        C = A * B
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])

    C = [[0.0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matvec(A: list[list[float]], v: list[float]) -> list[float]:
    """
    Calculate matrix-vector multiplication: 
        Ax = [A[0]*x, ..., A[m]*x]
    
    Args:
        A: input matrix
        v: input vector
    Returns:
        list: matrix-vector product
    """
    m = len(A)
    n = len(A[0])

    if n != len(v):
        raise ValueError("Matrix và vector phải có kích thước tương thích")

    result = []
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i][j] * v[j]
        result.append(s)
    
    return result

def identity_matrix(n: int) -> list[list[float]]:
    """
    Create identity matrix:
        I_n = [δ_{ij}]
    
    Args:
        n: matrix size
    Returns:
        list: identity matrix with size n x n
    """

    I = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        I[i][i] = 1.0
    return I

