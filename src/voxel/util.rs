use crate::voxel::chunk::{CHUNK_SIZE, CHUNK_VOLUME, Chunk, ChunkData};

pub const BOTTOM_NODE: u32 = u32::MAX;
pub const FULL_NODE: u32 = u32::MAX-1; 
pub const SVO_DEPTH: u32 = 5;
pub const TOTAL_SIZE: u32 = 1 << (SVO_DEPTH * 2);

pub fn offset_to_index(offset: vek::Vec3<usize>, size: usize) -> usize {
    assert!(offset.cmpge(&vek::Vec3::broadcast(0)).reduce_and());
    assert!(offset.cmplt(&vek::Vec3::broadcast(size)).reduce_and());
    
    offset.x + offset.y * size + offset.z * size * size
}

pub fn index_to_offset(index: usize, size: usize) -> vek::Vec3<usize> {
    assert!(index < (size*size*size));
    
    let x: usize = index % size;
    let y = (index / size) % size;
    let z = index / (size*size);
    vek::Vec3::new(x,y,z)
}

pub fn child_offset_to_child_index(offset: vek::Vec3<usize>) -> usize {
    offset_to_index(offset, 4)
}

pub fn child_index_to_child_offset(index: usize) -> vek::Vec3<usize> {
    index_to_offset(index, 4)
}

#[test]
fn test_indexing_stuff() {
    for index in 0..(64*64*64usize) {
        let offset = child_index_to_child_offset(index);
        let index2 = child_offset_to_child_index(offset);

        assert_eq!(index, index2);
    }
}

// hard-coded chunk granularity of 64x64x64
pub struct SparseImageChunk {
    pub origin: vek::Vec3<u32>,
    pub data: Vec<u8>,
    pub full: bool,
}

// contains PARTIAL data. NOT full NOR empty
pub struct PartialSparseImageChunk {
    pub origin: vek::Vec3<u32>,
}

/*
struct SimpleTraversalNode<'a> {
    node: &'a FlatNode,
    height: u32,
    origin: vek::Vec3<u32>,
}

pub fn convert_single_chunk_node_to_sparse_image_binding_chunk(origin: vek::Vec3<u32>, node: &FlatNode, nodes: &[FlatNode]) -> SparseImageChunk {
    let mut queue = VecDeque::<SimpleTraversalNode>::new();
    queue.push_back(SimpleTraversalNode { node, height: 3, origin }); // 4^3 = 64

    let mut data: Vec<u8> = vec![0u8; 64*64*64];

    let chunk_origin = origin;

    while let Some(SimpleTraversalNode { node, height, origin  }) = queue.pop_front() {
        let size: u32 = 4u32.pow(height);

        let pixel_coordinate_in_chunk = origin - chunk_origin;

        if height == 0 {
            // height could either be 0 (i.e single voxel) or 1 (i.e brick of 4x4x4)
            if height == 0 {
                let global = pixel_coordinate_in_chunk.as_::<usize>();
                let i = global.x + global.y * 64 + global.z * 64 * 64;
                data[i] = 255;
            } else if height == 1 {
                let mut bitmask = node.children.as_ref().map(|children| children.iter()
                    .enumerate()
                    .filter_map(|(i, x)| x.as_ref().map(|_| i))
                    .fold(0u64, |prev, i| (1u64 << i) | prev)
                ).unwrap_or_default();

                if node.children.is_none() {
                    bitmask = u64::MAX;
                }

                for x in 0..4 { 
                    for y in 0..4 { 
                        for z in 0..4 {
                            let global = vek::Vec3::new(x,y,z) + pixel_coordinate_in_chunk; 
                            let i = global.x + global.y * 64 + global.z * 64 * 64;
                            let bit_index = child_offset_to_child_index(vek::Vec3::new(x,y,z).as_::<usize>());

                            if is_set(bitmask, bit_index as u32) {
                                data[i as usize] = 255;
                            }
                        }
                    }    
                }
            }
        } else {
            if node.full {
                for x in 0..size { 
                    for y in 0..size { 
                        for z in 0..size {
                            let global = vek::Vec3::new(x,y,z) + pixel_coordinate_in_chunk; 
                            let i = global.x + global.y * 64 + global.z * 64 * 64;
                            data[i as usize] = 255;
                        }
                    }    
                }
            } else {
                // node is not full, we must compute bitmask of children and stuff
                if let Some(children) = node.children.as_ref() {
                    for (ci, child) in children.iter().enumerate().filter_map(|(ci, x)| x.as_ref().map(|x| (ci, x))) {
                        queue.push_back(SimpleTraversalNode { node: &nodes[*child], height: height - 1, origin: origin + child_index_to_child_offset(ci).as_::<u32>() * (size/4) });
                    }
                }
            }
        }
    }

    SparseImageChunk {
        origin,
        data,
        full: false,
    }
}

// hard-coded chunk granularity of 64x64x64
pub fn convert_to_sparse_image_chunks(nodes: &[FlatNode]) -> Vec<SparseImageChunk> {
    let mut queue = VecDeque::<SimpleTraversalNode>::new();
    queue.push_back(SimpleTraversalNode { node: &nodes[0], height: SVO_DEPTH, origin: vek::Vec3::zero() });

    let mut binding_chunks = Vec::<SparseImageChunk>::new();

    let mut num = 0;
    while let Some(SimpleTraversalNode { node, height, origin  }) = queue.pop_front() {
        //log::info!("h:{height}, o:{origin}");

        // chunks are handled differently. we convert...
        if height == 3 {
            num += 1;
            binding_chunks.push(convert_single_chunk_node_to_sparse_image_binding_chunk(origin, node, nodes));
            continue;
        }

        let size: u32 = 4u32.pow(height);

        if height == 0 {
            // should never reach this???
            panic!();
        } else {
            if node.full {
                // FIXME
                // figure out the space that this node spans, and add the single chunk nodes to accomodate
                // yes this is very stupid. not so sparse now bruh. wtv for now
                let num_chunks_size = size / 64;
                for x in 0..num_chunks_size { 
                    for y in 0..num_chunks_size { 
                        for z in 0..num_chunks_size { 
                            let chunk_pos = vek::Vec3::new(x,y,z) * 64 + origin;
                            let full = SparseImageChunk { origin: chunk_pos, data: Vec::new(), full: true };
                            binding_chunks.push(full);
                        }
                    }    
                }
            } else {
                // node is not full, we must compute bitmask of children and stuff
                if let Some(children) = node.children.as_ref() {
                    for (ci, child) in children.iter().enumerate().filter_map(|(ci, x)| x.as_ref().map(|x| (ci, x))) {
                        queue.push_back(SimpleTraversalNode {
                            node: &nodes[*child],
                            height: height - 1,
                            origin: origin + child_index_to_child_offset(ci).as_::<u32>() * (size/4)
                        });
                    }
                }
            }
        }
    }
    
    dbg!(num);

    binding_chunks
}
*/


// hard-coded chunk granularity of 64x64x64
pub fn convert_to_sparse_image_chunks(chunks: &[Chunk]) -> Vec<SparseImageChunk> {
    log::debug!("converting normal chunks to sparse image chunks... ");
    let mut binding_chunks = Vec::<SparseImageChunk>::new();

    for chunk in chunks {
        let data = match &chunk.voxel_data {
            ChunkData::Full => Vec::new(),
            ChunkData::Empty => Vec::new(),
            ChunkData::Partial(fixed_bit_set) => {
                let mut data = Vec::<vek::Vec4<u8>>::with_capacity(CHUNK_VOLUME);
                for bit_index in 0..CHUNK_VOLUME {
                    let is_set = fixed_bit_set.contains(bit_index);
                    data.push(if is_set { vek::Vec4::broadcast(255) } else { vek::Vec4::zero() });
                }
                data
            },
        };
        
        binding_chunks.push(SparseImageChunk {
            origin: chunk.position * (CHUNK_SIZE as u32),
            data: bytemuck::cast_slice(&data).to_vec(),
            full: chunk.is_full()
        });
    }

    log::debug!("finished with {} sparse image binding chunks", binding_chunks.len());

    binding_chunks
}