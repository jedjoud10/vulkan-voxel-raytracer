use fixedbitset::FixedBitSet;
use smallvec::SmallVec;

use crate::voxel::sparse::FlatNode;

pub enum ChunkData {
    Full,
    Empty,
    Partial(FixedBitSet)
}

pub struct Chunk {
    pub position: vek::Vec3<u32>,
    pub voxel_data: ChunkData,
    pub sparse_representation: Vec<FlatNode>,
}

impl Chunk {
    pub fn new(position: vek::Vec3<u32>, data: FixedBitSet) -> Self {
        let voxel_data = if data.is_full() {
            ChunkData::Full
        } else if data.is_clear() {
            ChunkData::Empty
        } else {
            ChunkData::Partial(data)
        };

        Self {
            position,
            voxel_data,
            sparse_representation: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match &self.voxel_data {
            ChunkData::Full => false,
            ChunkData::Empty => true,
            ChunkData::Partial(fixed_bit_set) => {
                let res = fixed_bit_set.is_clear();

                if res {
                    log::warn!("chunk was empty, but is not in special 'empty' state");
                }

                res
            },
        }
    }
    
    pub fn get(&self, i: usize) -> bool {
        match &self.voxel_data {
            ChunkData::Full => true,
            ChunkData::Empty => false,
            ChunkData::Partial(fixed_bit_set) => fixed_bit_set[i],
        }
    }

    pub fn rebuild(&mut self) {
        let bounds = vek::Aabb::<u32> {
            min: self.position * 64,
            max: self.position * 64 + 64,
        };

        let k = match &self.voxel_data {
            ChunkData::Full | ChunkData::Partial(_) => vec![FlatNode { bounds, children: None, full: true }],
            ChunkData::Empty => vec![FlatNode { bounds, children: None, full: false }],
        };
        self.sparse_representation = k;

        //self.sparse_representation = chunk_to_sparse(&self.voxel_data, self.position, bounds);
    }
}


fn chunk_to_sparse(data: &ChunkData, chunk_position: vek::Vec3<u32>, bounds: vek::Aabb<u32>) -> Vec<FlatNode> {
    let data = match data {
        ChunkData::Full => return vec![FlatNode { bounds, children: None, full: true }],
        ChunkData::Empty => return vec![FlatNode { bounds, children: None, full: false }],
        ChunkData::Partial(data) => data 
    };
    
    // because bread tastes better than key
    // no actually, because 4^3 = 64
    const CHUNK_64_HEIGHT_4_TREE: u32 = 3;

    // bottom up approach: do multiple passes, starting from the bottom
    // this will generate the "mips" of the chunk
    // this can be parallelized
    // this can be optimized further if using morton encoding, because then, groups of 4x4x4 nodes are a contiguous slice of 64 bits. We can use batch operations to speed that up instead of keeping the inner most 3 loops 
    let mut any_mip = data;
    let mut all_mip = data;

    let mut any_mips = [const { FixedBitSet::new() }; CHUNK_64_HEIGHT_4_TREE as usize];
    let mut all_mips = [const { FixedBitSet::new() }; CHUNK_64_HEIGHT_4_TREE as usize];
    
    for pass in 0..3 {
        let mip_size = 64 / (1 << ((1+pass)*2)); // 16, 4, 1

        let mut next_any_mip = FixedBitSet::with_capacity((mip_size as usize).pow(3));
        let mut next_all_mip = FixedBitSet::with_capacity((mip_size as usize).pow(3));

        for x in 0..mip_size {
            for y in 0..mip_size {
                for z in 0..mip_size {
                    let mut any = false;
                    let mut all = true;

                    for local_x in 0..4 {
                        for local_y in 0..4 {
                            for local_z in 0..4 {
                                let offset = vek::Vec3::new(x,y,z);
                                let local_offset = vek::Vec3::new(local_x, local_y, local_z);
                                let position: vek::Vec3<i32> = local_offset * 4 + offset;
                                let i = (position.x + position.y * 4 + position.z * 4 * 4) as usize; 
                                any |= any_mip[i];
                                all &= all_mip[i];
                            }
                        }
                    }

                    let i = (x + y * 4 + z * 4 * 4) as usize; 
                    next_any_mip.set(i, any);
                    next_all_mip.set(i, all);

                }
            }
        }

        any_mips[pass] = next_any_mip;
        all_mips[pass] = next_all_mip;

        any_mip = &any_mips[pass];
        all_mip = &all_mips[pass]; 

    }

    // start top down and create some nodes
    // we can inline the nodes in any fashion we want in the array, as long as their indices match up
    // we can write them in BFS or DFS order. does not matter
    super::util::convert_mips_to_nodes(chunk_position * 64, CHUNK_64_HEIGHT_4_TREE, &all_mips, &any_mips)
}
