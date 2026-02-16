# Voxel Raytracing Project (Vulkan, Rust, Slang)

## Screenies
<img width="1919" height="1197" alt="Screenshot 2026-01-09 210057" src="https://github.com/user-attachments/assets/16adee94-59e7-48e8-8e15-5ba4c42ac1da" />
<img width="1919" height="1199" alt="Screenshot 2026-01-09 210150" src="https://github.com/user-attachments/assets/4423947e-7894-45b2-9f88-8550984369ae" />


## Features
- PBR rendering (code copied from cflake engine and adapted to Slang)
- Realistic Sky Rendering using `SkyTheDragon`'s [sky atmo](https://www.shadertoy.com/view/t3XBWH) and `TheNuclearWolf`'s [fast-sky](https://www.shadertoy.com/view/lcGfDK
)
- World stored as a single 3D texture containing voxels.
- Naive 3D voxel ray-tracing using DDA and compute shaders 
- Custom "UV Unwrapping" by creating surfaces and unwrapping them.
    - Implemented using an extra 3D texture that contains **indices** of each voxel to an extra **buffer** that contains the color information (data type: ``uint8_4``, 4 bytes)
    - Allows for *temporal* effects / smoothing, without camera smearing (as the space isn't only the view space of the camera, it is the *entire world as a whole*) 
    - Currently only supports shadow / soft shadows. Tried experimenting with naive-GI but did not work very nicely (also was very expensive)
- Ticking logic system, separate from frame-based logic. Allows us to run the ``update`` compute shader periodically instead of every frame.


## Things I Tried
- Implement *octree* / *BHV* as a basic acceleration structure.
    - Got it working through multiple implementations:
        - Naive Octree Traversal: Just checks each node's 8 children using an AABB test...
        - DDA Recursive (stackless): Uses DDA to speed things up and recurses by calling the function itself. **Fastest one so far**
        - DDA Recursive (stack): Uses DDA to speed things up and stores results in an intermediate stack to be handled next iteration.
- Implement *sparse voxel octree* using buffer
    - Instead of storing the octree as mips inside a texture, we instead store a *sparse octree* inside a buffer
    - This *requires* us to recurse through the structure unfortunately, but it leads to much lower memory usage and we can use 64 bit brickmap logic to accelerate DDA as well
    - Again, this is problematic due to high VGPR pressure, which hurts occupancy. Some buffery latency is not able to be fully hidden away because of this
- Micro-Voxels: Implemented by storing a ```u64``` inside the voxel texture, which allows us to run a DDA on for sub-meter voxels.
- DDA "pre-computation" buffer: precomputes all the possible ```u64``` bitmasks on the CPU and uploads them to the GPU so that instead of doing "micro-DDA" we can just do a bitwise ```AND``` and check if there is an intersection between the ray and the micro-voxels. Works, but is *not* faster than just naive DDA. This is due to many reasons:
    - This is how it works:
        - It assumues the camera ray is coming from outside the block *towards* it
        - It bakes the micro-voxels that are "traversed" between the start point and end point of the ray-intersection test.
        - It does this by subdividing each face of the unit cube into 16*16 "pixels"
        - During baking, it loops over every face, checking what are all the possible intersections that could occur to every *other* face
        - It does this for 2 faces at a time, for each of their segments (so 4 nested loops for the segments and an extra loop for face pairs)
        - It keeps track of the "traversed" micro-voxels by tagging them in a ```u64```
        - At runtime, we look up the correct baked bitmask and ```AND``` the current fetched bitmask to check if there is a "possible" intersection (the baked intersections are *not* conservative) 

    - Bad latency hiding: we are currently bandwidth limited because do so many texture / buffer fetches and not many ALU operations. We can't hide latency that well
    - Bad occupancy: due to recursive octree traversal, occupancy is in shambles (4/16 on my 780m).
    - Computing the unique index for the buffer is very expensive: I don't know how to improve this. I need a better encoding scheme than just ``` ray enter face + ray exit face + enter face segment + exit face segment ```

## TODO
- Implement multiple chunk rendering
    - *Tip For Shadows*: At far enough distances, the per-voxel face texels will occupy less than one pixel on the screen. This means that it would be cheaper to do shadows for *each pixel*, instead of for *each voxel face texel*. 
- Experiment with dedicated Raytracing extensions. Maybe we could speed things up by using RT acceleration structures / queries?
- Implement a way to invoke multiple rays from the same ray. Will be implemented like this:
    1. Start with 1 ray, starting at the camera, at the specificed direction from screen-space UVs
    2. Compute shader sorts rays that are "close" to each other (by position and direction)
    3. Another compute shader runs the ray-tracing, stores the result in an extra buffer
    4. Fetch extra rays if there are any from the previous step. Add back to original buffer somehow (maybe double buffered?)
    5. Go back to 2. and re-execute until we run out of iterations or we run out of rays
- This could allow for less noisy reflections without reverting to temporal accumulation / smoothing, but it would be **heaps order** more expensive.
- Currently, the update shader exits immediately when we encounter a voxel with no valid surface, and it then starts multiple loops, where the outer one iterates over the 6 faces of each voxel. We could split this into two parts to improve performance (IMPLEMENTED!!!! WORKS!!!)
    1. Have one compute shader create the visible cube-facets. Does no DDA computation by itself.
    2. Invoke another shader *indirectly* to do the expensive per-face computations for each face, where each invocation is a separate face.
- This *should* improve performance since now we can use the threads more efficiently. Must profile first to see if my guess is correct (thread usage is low).
- Some sort of water simulation using SPH and sub-meter voxels