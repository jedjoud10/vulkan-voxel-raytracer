use std::collections::HashMap;

use ash::vk;
use gpu_allocator::vulkan::Allocator;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{buffer::Buffer, voxel::{chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk}, util::{index_to_offset, offset_to_index, try_offset_to_index}}};


#[derive(Debug)]
pub struct SingleChunkInBuffer {
    pub index_count: usize,
    pub vertex_start_offset: usize,
    pub first_index: usize,
}

pub struct VoxelMeshBuffers {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer, 
    pub chunks: Vec<SingleChunkInBuffer>,
}
impl VoxelMeshBuffers {
    pub unsafe fn new(chunks: &[super::chunk::Chunk], device: &ash::Device, pool: vk::CommandPool, queue: vk::Queue, allocator: &mut Allocator, binder: &Option<ash::ext::debug_utils::Device>) -> Self {
        let other_chunks: HashMap<vek::Vec3::<u32>, &Chunk> = chunks.iter().map(|c| (c.position, c)).collect::<HashMap<_, _>>();
        
        
        let everything = chunks.par_iter().map(|c| mesh_chunk(c, &other_chunks)).collect::<Vec<_>>();

        let mut vertices = Vec::<vek::Vec3<f32>>::new();
        let mut indices = Vec::<u32>::new();
        let mut chunks = Vec::<SingleChunkInBuffer>::new();

        for (single_chunk_vertices, single_chunk_indices) in everything {
            if single_chunk_indices.len() > 0 && single_chunk_vertices.len() > 0 {
                let vertex_start_offset = vertices.len();
                let index_count = single_chunk_indices.len();
                let first_index = indices.len();
                
                vertices.extend(single_chunk_vertices);
                indices.extend(single_chunk_indices);

                chunks.push(SingleChunkInBuffer { index_count, vertex_start_offset, first_index });
            }
        }

        for k in &chunks {
            log::debug!("chunk: {:?}", k);
        }

        let vertex_buffer = crate::buffer::create_buffer(device, allocator, vertices.len() * size_of::<vek::Vec3::<f32>>(), binder, "voxel chunk mesh vertex buffer", vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        crate::buffer::write_to_buffer(device, pool, queue, vertex_buffer.buffer, allocator, bytemuck::cast_slice(vertices.as_slice()));


        let index_buffer = crate::buffer::create_buffer(device, allocator, indices.len() * size_of::<u32>(), binder, "voxel chunk mesh index buffer", vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);
        crate::buffer::write_to_buffer(device, pool, queue, index_buffer.buffer, allocator, bytemuck::cast_slice(indices.as_slice()));


        Self {
            vertex_buffer,
            index_buffer,
            chunks,
        }
    }

    pub unsafe fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.vertex_buffer.destroy(device, allocator);
        self.index_buffer.destroy(device, allocator);
    }
}

const INDEX_QUAD_ORDER: [usize; 6] = [0, 1, 2, 2, 1, 3];
const INDEX_OPPOSITE_QUAD_ORDER: [usize; 6] = [1, 0, 2, 1, 2, 3];

// stupid and unoptimized but works for now ig
fn mesh_chunk(chunk: &Chunk, other_chunks: &HashMap<vek::Vec3::<u32>, &Chunk>) -> (Vec::<vek::Vec3<f32>>, Vec<u32>) {
    let bitset= match &chunk.voxel_data {
        super::chunk::ChunkData::Full => return (Vec::new(), Vec::new()),
        super::chunk::ChunkData::Empty => return (Vec::new(), Vec::new()),
        super::chunk::ChunkData::Partial(fixed_bit_set) => fixed_bit_set,
    };

    let mut vertices = Vec::<vek::Vec3<f32>>::new();
    let mut indices = Vec::<u32>::new();
    let mut lookup = vec![0u32; CHUNK_VOLUME];

    // first pass will generate vertex positions
    for x in 0..(CHUNK_SIZE-1) {
        for y in 0..(CHUNK_SIZE-1) {
            for z in 0..(CHUNK_SIZE-1) {
                let pos = vek::Vec3::new(x,y,z);

                let mut bitmask = 0u8;
                for neighbour in 0..8usize {
                    let neighbour_offset = index_to_offset(neighbour, 2);
                    let next_cell_is_set = bitset.contains(offset_to_index(pos + neighbour_offset, CHUNK_SIZE));

                    if next_cell_is_set {
                        bitmask |= 1u8 << neighbour as u8;
                    }
                }

                if bitmask != 0 && bitmask != 0xFF {
                    let vertex_position = pos.as_::<f32>() + 0.5f32 + chunk.position.as_::<f32>() * CHUNK_SIZE as f32;
                    let index = vertices.len();
                    lookup[offset_to_index(pos, CHUNK_SIZE)] = index as u32;
                    vertices.push(vertex_position);
                }
            }
        }
    }

    // second pass will generate quads
    for x in 1..(CHUNK_SIZE-1) {
        for y in 1..(CHUNK_SIZE-1) {
            for z in 1..(CHUNK_SIZE-1) {
                let pos = vek::Vec3::new(x,y,z);
                let is_set = bitset.contains(offset_to_index(pos, CHUNK_SIZE));
                
                for axis in 0..3 {
                    // next cell is the cell "forward" to the current cell based on axiss
                    let mut next_cell = pos;
                    next_cell[axis] += 1;
                    
                    let next_cell_is_set = bitset.contains(offset_to_index(next_cell, CHUNK_SIZE));
                    
                    if is_set != next_cell_is_set {
                        let mut quad_vertex_indices: [u32; 4] = [u32::MAX; 4];
                        
                        let dir = is_set ^ (axis == 1);
                        let vertex_offsets: [vek::Vec3<usize>; 4] = quad_vertex_offsets_for_axis(axis as u32);

                        // inside quad local vertex index
                        for index in 0..4 {
                            let target = vertex_offsets[index] + next_cell - 1;

                            if let Some(looked_up_vertex_index) = try_offset_to_index(target, CHUNK_SIZE).map(|x| lookup[x]) {
                                quad_vertex_indices[index] = looked_up_vertex_index;
                            }
                        }

                        // don't do anything if quad contains invalid vertices
                        if quad_vertex_indices.iter().any(|x| *x == u32::MAX) {
                            continue;
                        }

                        // holy cursed
                        let quad_vertex_order = if dir { INDEX_QUAD_ORDER } else { INDEX_OPPOSITE_QUAD_ORDER };
                        for what_to_call_this_index in 0..6 {
                            indices.push(quad_vertex_indices[quad_vertex_order[what_to_call_this_index]]);
                        }
                    }
                }
            }
        }
    }

    return (vertices, indices);
}

fn quad_vertex_offsets_for_axis(axis: u32) -> [vek::Vec3<usize>; 4] {
    match axis {
        0 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(0, 1, 0), vek::Vec3::new(0, 0, 1), vek::Vec3::new(0, 1, 1)], // x
        1 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(1, 0, 0), vek::Vec3::new(0, 0, 1), vek::Vec3::new(1, 0, 1)], // y
        2 => [vek::Vec3::new(0, 0, 0), vek::Vec3::new(1, 0, 0), vek::Vec3::new(0, 1, 0), vek::Vec3::new(1, 1, 0)], // z
        _ => unreachable!()
    }
}