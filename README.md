# Image Topologies Representation using Graphs

Visualize the topology of any RGB image with the help of graphs. The program converts any image
you provide to a depth map and extracts a graph topology. The edges represent a smooth gradient 
between adjacent pixels, where each pixel is represented as a node in the graph. 

<div style="display: flex;">
    <img src="https://github.com/AD-lite24/Image-Topology-Visualizer/assets/96363931/05ae0bab-a04f-4955-bc26-9de3698f4a8a" alt="person_graph_render" style="width: 50%;">
    <img src="https://github.com/AD-lite24/Image-Topology-Visualizer/assets/96363931/0601587a-c31f-4619-847b-15d50faad75c" alt="person_test" style="width: 50%;">
</div>

## Features

* The graph computation and rendering is cuda accelerated allowing for almost instantaneous rendering of even very large graphs, often exceed 50k nodes. 

* The nodes in the graph are also color coded according to the depth of their corresponding pixel, allowing for easy visualization and another dimension of understanding to the generation

* Tweaking the model size to get graphs from coarser to smoother depth maps. 

* Uses the zero shot trained model Intel Midas and OpenGL for rendering. 

* Graph extraction parallelized using torch while graph node positions computed parallely during rendering runtime manually. Rendering time complexity is O(V + E) where V is number of vertices and E is number of edges.


