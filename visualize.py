from numba import cuda
import numpy as  np
import warnings
import OpenGL.GL as gl


@cuda.jit
def compute_positions_kernel(positions, N):
    idx = cuda.grid(1)
    if idx < N:

        positions[idx, 0] = idx * 10.0
        positions[idx, 1] = idx * 10.0

adjacency_matrix = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
])

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glLoadIdentity()

    gl.glBegin(gl.GL_POINTS)
    for i in range(N):
        gl.glVertex2f(node_positions[i, 0], node_positions[i, 1])
    gl.glEnd()

    glut.glutSwapBuffers()

N = adjacency_matrix.shape[0]

node_positions = np.zeros((N, 2), dtype=np.float32)
node_positions_gpu = cuda.to_device(node_positions)

block_size = 256
grid_size = (N + block_size - 1) // block_size
compute_positions_kernel[grid_size, block_size](node_positions_gpu, N)

# copy back to cpu
node_positions_gpu.copy_to_host(node_positions)
print(node_positions)


