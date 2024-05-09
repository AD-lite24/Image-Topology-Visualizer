from numba import cuda
import OpenGL.GLUT as glut
import OpenGL.GL as gl
import OpenGL.GLU as glu
import torch
import matplotlib.cm as cm
import cv2
from utilities import ExtractGraph, Encoder,  MaxPool
import OpenGL.GL.framebufferobjects as glfbo
import numpy as np
from tqdm import tqdm

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def visualize(edge_index, rows, cols, depth_vals, save, save_name):
    
    @cuda.jit
    def compute_positions_kernel(positions, N, cols):
        idx = cuda.grid(1)
        if idx < N:
            positions[idx, 0] = idx % cols # x
            positions[idx, 1] = idx // cols # y 


    def init(rows, cols):
        gl.glClearColor(0.0, 0.0, 0.0, 0.0) # bg color black
        glu.gluOrtho2D(-2, cols + 4, -2, rows + 4)


    def drawPoints():
        
        if save:
            width = glut.glutGet(glut.GLUT_WINDOW_WIDTH)
            height = glut.glutGet(glut.GLUT_WINDOW_HEIGHT)

            fbo = glfbo.glGenFramebuffers(1)
            glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, fbo)

            color_buffer = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, color_buffer)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
            glfbo.glFramebufferTexture2D(glfbo.GL_FRAMEBUFFER, glfbo.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, color_buffer, 0)

            assert glfbo.glCheckFramebufferStatus(glfbo.GL_FRAMEBUFFER) == glfbo.GL_FRAMEBUFFER_COMPLETE
            
            glfbo.glBindFramebuffer(glfbo.GL_FRAMEBUFFER, fbo)


        gl.glClear(gl.GL_COLOR_BUFFER_BIT) 
        colormap = cm.get_cmap('viridis')

        gl.glPointSize(5.0)  # node point size
        gl.glBegin(gl.GL_POINTS)
        depth = np.array(depth_vals.squeeze().cpu())

        low_percentile = np.percentile(depth, 5)
        high_percentile = np.percentile(depth, 95)

        for i in tqdm(range(len(node_positions))):
            x_p = int(node_positions[i, 0])
            y_p = int(node_positions[i, 1])

            # note that x is cols and y is rows, and index 0 is rows
            val = depth[y_p, x_p]
            
            # to fix for opengl coordinate system where (0,0) is at the 
            # bottom left
            y_p = depth.shape[0] - 1 - y_p
            normalized_val = normalize(val, low_percentile, high_percentile)

            color = colormap(normalized_val)
            # print(val, color)
            gl.glColor3f(color[0], color[1], color[2])
            gl.glVertex2f(x_p, y_p)
        gl.glEnd()
        
        gl.glColor3f(1.0, 1.0, 1.0) 
        gl.glLineWidth(1.4)  
        gl.glBegin(gl.GL_LINES)

        for i in tqdm(range(len(edge_index[0]))):
            n1, n2 = edge_index[0, i], edge_index[1, i] 
            n1_x, n1_y = node_positions[n1, 0], node_positions[n1, 1]
            n2_x, n2_y = node_positions[n2, 0], node_positions[n2, 1]

            # again need to correct for y coordinate since opengl coordinate
            # system is with (0, 0) at bottom left instead of top left
            n1_y = rows - 1 - n1_y
            n2_y = rows - 1 - n2_y

            gl.glVertex2f(n1_x, n1_y)
            gl.glVertex2f(n2_x, n2_y)

        gl.glEnd()
        gl.glFlush() 

        if save:
            rendered_image = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            rendered_image = np.frombuffer(rendered_image, dtype=np.uint8).reshape((height, width, 3))
            rendered_image_flipped = cv2.flip(rendered_image, 0)

            cv2.imwrite(f"outputs/{save_name}_out_save_render.png", cv2.cvtColor(rendered_image_flipped, cv2.COLOR_RGB2BGR))

            glfbo.glDeleteFramebuffers(1, [fbo])
            gl.glDeleteTextures(1, [color_buffer])

    
    N = rows*cols # number of nodes 
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    node_positions_gpu = cuda.device_array((N, 2), dtype=np.float32)
    compute_positions_kernel[grid_size, block_size](node_positions_gpu, N, cols)
    node_positions = node_positions_gpu.copy_to_host()

    glut.glutInit() 
    glut.glutInitDisplayMode(glut.GLUT_SINGLE | glut.GLUT_RGB) 
    glut.glutInitWindowSize(cols*15, rows*15) 
    glut.glutInitWindowPosition(100, 100) 
    glut.glutCreateWindow(b"Graph Topology")
    init(rows, cols) 
    glut.glutDisplayFunc(drawPoints) # Set the display callback function
    glut.glutMainLoop() 
    

def initialize_midas(device, midas_model_type):

    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if midas_model_type == "DPT_Large" or midas_model_type == "DPT_Hybrid":
        midas_transform = midas_transforms.dpt_transform
    else:
        midas_transform = midas_transforms.small_transform

    return midas, midas_transform


def main(rgb_path : str, device : object, model_type : str, save : bool):

    midas, transform = initialize_midas(device=device, midas_model_type=model_type)

    name = rgb_path.split('/')[-1].split('.')[0]

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        name += "_highres"

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    input_rgb = transform(rgb).to(device=device)

    print("Shape after transform (N, C, H, W) ", input_rgb.shape)

    with torch.no_grad():
        prediction = midas(input_rgb)
    
    depth_map = prediction
    print("Depth map shape ", depth_map.shape)
    rows, cols = depth_map.shape[1:]

    pool = MaxPool(pool_size=2)
    # pooled_rgb = pool.forward(input_rgb)
    pooled_depth = pool.forward(depth_map)
    print("Pooled depth shape: ", pooled_depth.shape)

    rows, cols = rows//2, cols//2 # gets pooled to half later
    print("rows: ", rows, " cols: ", cols)
    extractor = ExtractGraph().to(device=device)
    encoder = Encoder().to(device=device)

    edge_index = extractor.forward(d_coarse=depth_map, R_scale=0.4)
    # adj_np = np.array(adj.detach().cpu())

    print("No. of edges to display ", edge_index.shape[1])
    visualize(edge_index, rows, cols, pooled_depth, save, name)
    
if __name__ == "__main__":
    
    input_file = "data/car_highway_test.jpeg"
    
    save = True
    model_ver = 2

    midas_type = {
        0 : "DPT_Large",
        1 : "DPT_Hybrid",
        2 : "MiDaS_small"
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    main(input_file, device, midas_type[model_ver], save)
