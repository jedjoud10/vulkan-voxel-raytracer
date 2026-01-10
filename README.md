# Voxel Raytracing Project (Vulkan, Rust, Slang)

## Screenies
<img width="1919" height="1197" alt="Screenshot 2026-01-09 210057" src="https://github.com/user-attachments/assets/16adee94-59e7-48e8-8e15-5ba4c42ac1da" />
<img width="1919" height="1199" alt="Screenshot 2026-01-09 210150" src="https://github.com/user-attachments/assets/4423947e-7894-45b2-9f88-8550984369ae" />


## Features
- World stored as a single 3D texture containing voxels.
- Naive 3D voxel ray-tracing using DDA and compute shaders 
- Custom "UV Unwrapping" by creating surfaces and unwrapping them.
    - Implemented using an extra 3D texture that contains **indices** of each voxel to an extra **buffer** that contains the color information (data type: ``uint8_4``, 4 bytes)
    - Allows for *temporal* effects / smoothing, without camera smearing (as the space isn't only the view space of the camera, it is the *entire world as a whole*) 
    - Currently only supports shadow / soft shadows. Tried experimenting with naive-GI but did not work very nicely (also was very expensive)
- Ticking logic system, separate from frame-based logic. Allows us to run the ``update`` compute shader periodically instead of every frame.

## TODO
- Implement *octree* / *BHV* as a basic acceleration structure.
    - I have tried before to implement *octrees* with DDA and have failed. Maybe this time it will work
- Experiment with dedicated Raytracing extensions. Maybe we could speed things up by using RT acceleration structures / queries?
- Implement a way to invoke multiple rays from the same ray. Will be implemented like this:
    1. Start with 1 ray, starting at the camera, at the specificed direction from screen-space UVs
    2. Compute shader sorts rays that are "close" to each other (by position and direction)
    3. Another compute shader runs the ray-tracing, stores the result in an extra buffer
    4. Fetch extra rays if there are any from the previous step. Add back to original buffer somehow (maybe double buffered?)
    5. Go back to 2. and re-execute until we run out of iterations or we run out of rays
- This could allow for less noisy reflections without reverting to temporal accumulation / smoothing, but it would be **heaps order** more expensive.
- Currently, the update shader exits immediately when we encounter a voxel with no valid surface, and it then starts multiple loops, where the outer one iterates over the 6 faces of each voxel. We could split this into two parts to improve performance
    1. Have one compute shader create the visible cube-facets. Does no DDA computation by itself.
    2. Invoke another shader *indirectly* to do the expensive per-face computations for each face, where each invocation is a separate face.
- This *should* improve performance since now we can use the threads more efficiently. Must profile first to see if my guess is correct (thread usage is low).
