use std::{cell::RefCell, collections::VecDeque, ffi::{CStr, CString}, num::NonZeroU32, rc::{Rc, Weak}, str::FromStr, time::Instant};
use crate::utils::*;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use smallvec::SmallVec;
use crate::{buffer::{self, Buffer}, pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};
use super::SVO_DEPTH;

pub struct RecursiveNode {
    pub children: Option<Box<[Option<Box<RecursiveNode>>; 64]>>,
    pub bottom: bool, // set to true when height=0 or height=1
    pub full: bool, // set to true when all underlying nodes are full. This avoids having to store them in memory. If this is set then we will discard the result of `children` for any computations if the ray hits this node
}

fn full_node() -> RecursiveNode {
    RecursiveNode {
        children: None,
        bottom: false,
        full: true,
    }
}

fn menger_like(base: u32, seed: u32) -> RecursiveNode {
    if base == 0 {
        return RecursiveNode {
            children: None,
            bottom: true,
            full: false,
        };
    }

    let mut children = [const { None }; 64];
    let mut recursive= SmallVec::<[u8; 8]>::new();

    //let height = pseudo_random(0x12AB ^ base ^ seed) % 4;

    for i in 0..12 {
        recursive.push((pseudo_random(0x12AB ^ base ^ seed ^ (i * 3215)) % 64) as u8);
    }

    /*
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let vec = vek::Vec3::new(x,y,z);
                let i = x + y * 4 + z * 4 * 4;
                let cmp = vec.cmpeq(&vek::Vec3::broadcast(0)) | vec.cmpeq(&vek::Vec3::broadcast(4));
                let a = cmp.map(|x| if x { 1 } else { 0 });
                let k: i32 = a.x << 1 | a.y << 2 | a.z << 3; 
                if k.count_ones() >= 2 && pseudo_random(0x12AB ^ base ^ seed ^ (i*321)) % 4 < 3 {
                    recursive.push(i as u8);
                }
            }
        }
    }
    */

    let generated_children = recursive.into_par_iter().map(|i| {
        let dst_index = (pseudo_random((*i as u32) ^ 0x03f23 ^ base ^ seed) % 64);
        let child = Box::new(menger_like(base - 1, dst_index));
        (i, child)
    }).collect::<Vec<_>>();

    for (dst_index, child) in generated_children {
        children[*dst_index as usize] = Some(child);
    }
    let full = children.iter().all(|child| child.as_ref().map(|c| c.full).unwrap_or_default());
    
    RecursiveNode {
        children: Some(Box::new(children)),
        bottom: base == 1,
        full: full, 
    }
}

fn plains(base: u32, seed: u32) -> RecursiveNode {
    if base == 0 {
        return RecursiveNode {
            children: None,
            bottom: true,
            full: false,
        };
    }

    let mut children = [const { None }; 64];
    let mut recursive= SmallVec::<[u8; 8]>::new();

    let height = pseudo_random(0x12AB ^ base ^ seed) % 4;

    for x in 0..4 {
        for y in 0..height {
            for z in 0..4 {
                let vec = vek::Vec3::new(x,y,z);
                let i = x + y * 4 + z * 4 * 4;

                if (y+1) < height {
                    children[i as usize] = Some(Box::new(full_node()));
                } else {
                    recursive.push(i as u8);
                }
            }
        }
    }

    let generated_children = recursive.into_par_iter().map(|i| {
        let dst_index = (pseudo_random((*i as u32) ^ 0x03f23 ^ base ^ seed) % 64);
        let child = Box::new(plains(base - 1, dst_index));
        (i, child)
    }).collect::<Vec<_>>();

    for (dst_index, child) in generated_children {
        children[*dst_index as usize] = Some(child);
    }
    let full = children.iter().all(|child| child.as_ref().map(|c| c.full).unwrap_or_default());
    
    RecursiveNode {
        children: Some(Box::new(children)),
        bottom: base == 1,
        full: full, 
    }
}

fn spike(base: u32, cx: usize, cz: usize, seed: u32) -> RecursiveNode {
    if base == 0 {
        return RecursiveNode {
            children: None,
            bottom: true,
            full: false,
        };
    }

    let mut children = [const { None }; 64];
    let mut recursive= SmallVec::<[(u8, usize, usize); 8]>::new();

    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let vec = vek::Vec3::new(x,y,z);
                let i = x + y * 4 + z * 4 * 4;

                if (pseudo_random(0x44AB ^ base ^ seed ^ (z as u32) ^ (x as u32 + 534243)) % 5 < 2) {
                    if y < 3 {
                        children[i] = Some(Box::new(full_node()));
                    } else {
                        if pseudo_random(0x44AB ^ base ^ seed + y as u32) % 5 < 2 {
                            children[i] = Some(Box::new(test_sparse_voxel_octree_recurse(base.saturating_sub(2), seed)));
                        } else {
                            recursive.push((i as u8, cx, cz));
                        }
                    }
                }
            }
        }
    }

    let generated_children = recursive.into_par_iter().map(|(i, cx, cz)| {
        let dst_index = (pseudo_random((*i as u32) ^ 0x03f23 ^ base ^ seed) % 64);
        let child = Box::new(spike(base - 1, *cx, *cz, dst_index));
        (i, child)
    }).collect::<Vec<_>>();

    for (dst_index, child) in generated_children {
        children[*dst_index as usize] = Some(child);
    }
    let full = children.iter().all(|child| child.as_ref().map(|c| c.full).unwrap_or_default());
    
    RecursiveNode {
        children: Some(Box::new(children)),
        bottom: base == 1,
        full: full, 
    }
}

fn test_sparse_voxel_octree_recurse(base: u32, seed: u32) -> RecursiveNode {
    if base == 0 {
        return RecursiveNode {
            children: None,
            bottom: true,
            full: false,
        };
    }

    let mut children = [const { None }; 64];

    // small chance to exit early
    if (pseudo_random(0xA23 ^ base ^ seed) % 10) < 2 {
        return RecursiveNode {
            children: None,
            bottom: false,
            full: false,
        }
    }

    let mut recursive= SmallVec::<[u8; 8]>::new();

    for x in  0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let vec = vek::Vec3::new(x,y,z);
                let i = x + y * 4 + z * 4 * 4;

                // base plate
                if y == 0 {
                    children[i] = Some(Box::new(full_node()));
                    //children[i] = Some(Box::new(base_plate(base - 1)));
                }

                if y == 1 {
                    if (x == 0 || x == 3) || (z == 0 || z == 3) {
                        // inductive case
                        recursive.push(i as u8);
                    } else {
                        // mid plate
                        //recursive.push(i as u8);
                        children[i] = Some(Box::new(full_node()));
                        //children[i] = Some(Box::new(base_plate(base - 1)));
                    }
                }

                if y == 2 {
                    if (x == 1 || x == 2) && (z == 1 || z == 2) {
                        // inductive case
                        if pseudo_random(0x44AB ^ base ^ seed) % 5 < 2 {
                            //children[i] = Some(Box::new(spike(base, x, z, seed)));
                            children[i] = Some(Box::new(menger_like(base, seed)));
                        } else {
                            recursive.push(i as u8);
                        }
                    }
                }
            }
        }
    }

    let generated_children = recursive.into_par_iter().map(|i| {
        let dst_index = (pseudo_random((*i as u32) ^ 0x03f23 ^ base ^ seed) % 64);
        let child = Box::new(test_sparse_voxel_octree_recurse(base - 1, dst_index));
        (i, child)
    }).collect::<Vec<_>>();

    for (dst_index, child) in generated_children {
        children[*dst_index as usize] = Some(child);
    }
    let full = children.iter().all(|child| child.as_ref().map(|c| c.full).unwrap_or_default());
    
    RecursiveNode {
        children: Some(Box::new(children)),
        bottom: base == 1,
        full: full, 
    }
}

pub fn create_recursive_structure() -> RecursiveNode {
    log::debug!("creating recursive structure of depth {SVO_DEPTH}...");
    test_sparse_voxel_octree_recurse(SVO_DEPTH, 0x0323f)
}