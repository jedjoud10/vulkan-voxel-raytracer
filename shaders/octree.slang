#ifndef OCTREE
#define OCTREE


// https://tavianator.com/2022/ray_box_boundary.html
bool ray_box_intersection(
    float3 inv_dir,
    float3 ray_pos,
    float3 aabb_min,
    float3 aabb_max,
    out float _tmin,
) {
    float tmin = 0.0, tmax = 999999;

    float3 t1 = (aabb_min - ray_pos) * inv_dir;
    float3 t2 = (aabb_max - ray_pos) * inv_dir;

    float3 _min = min(t1, t2);
    float3 _max = max(t1, t2);

    tmin = max(max3(_min.x, _min.y, _min.z), tmin);
    tmax = min(min3(_max.x, _max.y, _max.z), tmax);

    _tmin = tmin;

    return tmin < tmax;
}

typealias PackedNode = Node;

struct Node {
    static Node root(int size) {
        return Node(0, 0, 0, 0, size);
    }

    // position. between zero and SIZE
    uint3 position;

    // represented by 4 bits, so range is [0..15]
    // depth of the root node is zero
    uint depth_level;

    // distance that we can safely "skip" over to reach the node. always an lower bound to the actual distance
    // this is scaled by the voxel size of this node's voxel size
    // this is a value in the range [0..1]
    float distance;

    uint8_t3 next_min_bounds_world_space;
    uint8_t3 next_max_bounds_world_space;

    PackedNode into_raw() {
        return this;
        /*
        // pack the position in the first 3 bytes
        // which means that max map/chunk size IS limited to 256. oh well
        // needs chunking!
        PackedNode packed = position.x & 0xFF | ((position.y & 0xFF) << 8) | ((position.z & 0xFF) << 16);

        // pack the depth level as well. 4 bits
        packed |= (depth_level & 0b1111) << 24;

        // pack distance
        uint clamped = (uint)clamp(ceil(distance * 15), 0, 15);
        packed |= (clamped & 0b1111) << 28;

        return packed;
        */
    }

    static Node from_raw(PackedNode raw) {
        return raw;
        /*
        Node node;
        node.position = uint3(raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF);
        node.depth_level = (raw >> 24) & 0b1111;
        node.distance = ((raw >> 28) & 0b1111) / 15.0;
        return node;
        */
    }

}



/*
public static const int3[] CHILDREN_OFFSETS = {
    int3(0, 0, 0),
    int3(1, 0, 0),
    int3(0, 0, 1),
    int3(1, 0, 1),
    int3(0, 1, 0),
    int3(1, 1, 0),
    int3(0, 1, 1),
    int3(1, 1, 1),
};



bool traverse(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
    out float3 position,
    out uint3 block_pos,
    out float3 block_uvs,
    out int face,
    out int iterations,
) {
    iterations = 0;
    if (dda_with_face<8>(fetcher, ray_dir, ray_pos, position, block_pos, block_uvs, face)) {
        return true;
    } else {
        ray_pos += ray_dir * sqrt(8);
    }

    //return ray_box_intersection(ray_dir, ray_pos, -3, 3) ? 1 : 0;
    float3 inv_dir = 1 / ray_dir;
    float3 dir_sign = sign(ray_dir);
    Stack<12, uint> stack;
    stack.append(Node(0, 0, 2).into_raw(), 0); // add the root node 

    static const int HYBRID_LVL = 6; 
    uint raw_node;
    Node node;
    float node_tmin;
    while (stack.tryDequeue(raw_node, node_tmin)) {
        node = Node.from_raw(raw_node);
        uint8_t1 fetched = node.value;
        uint depth = node.depth_level;
        uint3 pos = node.position;
        
        // check if this node is a parent and it has children... 
        if ((fetched & 2) == 2 && depth < HYBRID_LVL) {
            // add the children to the stack
            for (int c = 0; c < 8; c++) {
                uint3 child_position = pos * 2 + CHILDREN_OFFSETS[c];
                int child_level = depth + 1;
                uint4 packed = uint4(child_position, child_level);

                int mip_scale = (1 << packed.w);
                float3 aabb_size = SIZE / mip_scale;
                float3 aabb_min = packed.xyz * aabb_size;
                float3 aabb_max = packed.xyz * aabb_size + aabb_size; 

                // if empty, skip...
                uint8_t1 child_fetched = fetcher.fetch_raw(packed.xyz, MIP_LEVELS - packed.w - 1);
                if (child_fetched != 0) {
                    // child must intersect ray...
                    float local_tmin = 0;
                    if (ray_box_intersection(inv_dir, ray_pos, aabb_min, aabb_max, local_tmin)) {
                        stack.append(Node(child_position, child_level, child_fetched).into_raw(), local_tmin);
                    }
                }
            }
        }

        // check if this node is a voxel leaf node...
        if (depth >= HYBRID_LVL) {
            position = ray_pos + (node_tmin - 0.1) * ray_dir;
            if (dda_with_face<1 << (MIP_LEVELS - HYBRID_LVL + 1)>(fetcher, ray_dir, ray_pos + (node_tmin - 0.1) * ray_dir, position, block_pos, block_uvs, face)) {
                return true;
            }
        }

        // check if this node is a voxel leaf node...
        if ((fetched & 1) == 1) {
            float tmp = 0;
            ray_box_intersection(inv_dir, ray_pos, pos, pos + 1, tmp);
            position = ray_pos + (tmp + 0.001) * ray_dir;
            dda_with_face<1>(fetcher, ray_dir, ray_pos + (tmp - 0.01) * ray_dir, face);
            return true;
        }

        iterations++;
    }

    return false;
}


int traverse2(
    Fetcher fetcher,
    float3 ray_dir,
    float3 ray_pos,
) {
    //return ray_box_intersection(ray_dir, ray_pos, -3, 3) ? 1 : 0;
    float3 inv_dir = 1 / ray_dir;
    Stack<2, uint> stack;
    stack.append(Node(0, 0, 2).into_raw(), 0); // add the root node 

    uint raw_node;
    Node node;
    float node_tmin;
    int iterations = 0;
    while (stack.tryDequeue(raw_node, node_tmin)) {
        node = Node.from_raw(raw_node);
        uint8_t1 fetched = node.value;
        uint depth = node.depth_level;
        uint3 pos = node.position;
        
        // check if this node is a parent and it has children... 
        if ((fetched & 2) == 2 && depth != (MIP_LEVELS - 1)) {
            int child_level = depth + 1;
            int mip_scale = (1 << child_level);
            int aabb_size = SIZE >> child_level;

            // add the children to the stack
            for (int c = 0; c < 8; c++) {
                uint3 child_position = pos * 2 + CHILDREN_OFFSETS[c];
                uint4 packed = uint4(child_position, child_level);

                float3 aabb_min = packed.xyz * aabb_size;
                float3 aabb_max = packed.xyz * aabb_size + aabb_size; 

                // if empty, skip...
                uint8_t1 child_fetched = fetcher.fetch_raw(packed.xyz, MIP_LEVELS - packed.w - 1);
                if (child_fetched != 0) {
                    // child must intersect ray...
                    float local_tmin = 0;
                    if (ray_box_intersection(inv_dir, ray_pos, aabb_min, aabb_max, local_tmin)) {
                        stack.append(Node(child_position, child_level, child_fetched).into_raw(), local_tmin);
                    }
                }
            }
        }

        // check if this node is a voxel leaf node...
        if ((fetched & 1) == 1) {
            return iterations;
        }

        iterations++;
    }

    return iterations;
}
*/

#endif