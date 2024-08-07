# Image Topologies Representation using Graphs

Visualize the topology of any RGB image with the help of graphs. The program converts any image
you provide to a depth map and extracts a graph topology. The edges represent a smooth gradient 
between adjacent pixels, where each pixel is represented as a node in the graph. 


| Original | Coarse Topology | Smooth Topology |
|---------|---------|---------|
| <img src="https://github.com/AD-lite24/Image-Topology-Visualizer/assets/96363931/0601587a-c31f-4619-847b-15d50faad75c" alt="person_test" height="300"> | <img src="https://github.com/AD-lite24/Image-Topology-Visualizer/assets/96363931/05ae0bab-a04f-4955-bc26-9de3698f4a8a" alt="person_graph_render_coarse" height="300"> | <img src="https://github.com/AD-lite24/Image-Topology-Visualizer/assets/96363931/d2125d22-d84e-46ff-a7f3-b36c45972a86" alt="person_graph_render_smooth" height="300" width="190">



## Features

* The graph computation and rendering is cuda accelerated allowing for almost instantaneous rendering of even very large graphs, often exceeding 50k nodes. 

* The nodes in the graph are color coded according to the estimated depth of the corresponding pixel, allowing for easy visualization and another dimension of understanding to the generation.

* Tweaking the model size to get graphs from coarser to smoother depth maps. 

* Uses the zero shot trained model Intel Midas and OpenGL for rendering. 

* Graph extraction parallelized using torch while graph node positions computed parallely during rendering runtime manually. Rendering time complexity is O(V + E) where V is number of vertices and E is number of edges.

## Usage

Run using 

`$ python3 main.py <path to input image>`

You can use the -t argument to set the model size to use smoother depths for graph extraction. Larger model -> higher resolution -> more pixels -> denser graph. Not necessarily desirable though. Values can be set to either 0, 1, 2; 2 being the smallest (coarsest) model and the default option

Use the -s flag to save your render instead of displaying. It will be stored in directory called outputs inside your current directory. 

Use the following command to turn off the numba low occupancy warning since our grid size won't really be big enough to warrant an underutilization warning.

`$ conda env config vars set NUMBA_CUDA_LOW_OCCUPANY_WARNINGS=0`

