use std::{ffi::{CStr, CString}, str::FromStr};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};
use vek::approx::AbsDiffEq;
use crate::{pipeline::{ComputePipeline, PushConstants2, VoxelGeneratePipeline, VoxelTickPipeline}};

// cube: 6 quads
// each quad: 4 segments = 2 bits
// ray intersection: input face + output face
// input face: number in [0..5]: repr 3 bits
// output face: number in [0..5]: repr 3 bits
// 3 + 3 + 2 + 2 = 10 bits

// 30 total possible pairs = 5 * 6
// pairs are symmetric, so (1, 2)=(2,1). de-duplicate by making fst < snd
// by deduplicating, we get a total of 15 pairs, representable by 4 bits only
// 4 + 4 + 4 = 12 bits !!!
// (0, 1)
// (0, 2)
// (0, 3)
// (0, 4)
// (0, 5)
// (1, 2)
// (1, 3)
// (1, 4)
// (1, 5)
// (2, 3)
// (2, 4)
// (2, 5)
// (3, 4)
// (3, 5)
// (4, 5)



fn dda_test(
    raw: u64,
    mask: u64,
    ray_pos: vek::Vec3<f32>,
    ray_dir: vek::Vec3<f32>,
) {
    let modified_ray_pos = ray_pos * 4.0;
    let mut floored_pos = modified_ray_pos.floor().as_::<i32>();
    let inv_dir = ray_dir.map(|x| x.abs().max(1e-5)).recip();
    let dir_sign = ray_dir.map(|x| x.signum());
    let mut side_dist = ((dir_sign * (floored_pos.as_::<f32>() - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);

    let min = vek::Vec3::<i32>::broadcast(0);
    let max = vek::Vec3::<i32>::broadcast(4);

    // check what the result is for the mask method
    let mask_pass = (raw & mask) != 0;

    let mut hit = false;
    let mut within_volume = false;
    for i in 0..12 {
        if floored_pos.partial_cmpge(&min).reduce_and() && floored_pos.partial_cmplt(&max).reduce_and() {
            within_volume = true;
            let local = floored_pos.map(|x| x % 4);
            let index = local.z + local.y * 4 + local.x * 4 * 4;
            assert!(index >= 0 && index < 64);

            if ((raw >> index) & 1 == 1) {
                hit = true;
                break;
            }
        }

        if (floored_pos.partial_cmplt(&min).reduce_or() || floored_pos.partial_cmpge(&max).reduce_or()) && within_volume {
            break;
        }

        let eqs = vek::Vec3::broadcast(side_dist.reduce_partial_min()).partial_cmpeq(&side_dist);
        let eqs = eqs.map(|x| if x { 1.0f32 } else { 0.0f32 });
        side_dist += eqs * inv_dir;
        floored_pos += (eqs * dir_sign).as_::<i32>();
    }

    // check if the mask method gives the same result as the naive method
    assert_eq!(mask_pass, hit);
    
}

const FACE_NORMALS: [vek::Vec3<f32>; 6] = [
    vek::Vec3::<f32>::new(-1f32, 0f32, 0f32),
    vek::Vec3::<f32>::new(0f32, -1f32, 0f32),
    vek::Vec3::<f32>::new(0f32, 0f32, -1f32),
    vek::Vec3::<f32>::new(1f32, 0f32, 0f32),
    vek::Vec3::<f32>::new(0f32, 1f32, 0f32),
    vek::Vec3::<f32>::new(0f32, 0f32, 1f32)
];

fn unflatten_uv_offset(face: usize, u: f32, v: f32) -> vek::Vec3<f32> {
    let axis = face % 3;
    let positive = face / 3 == 1;
    let normal = if positive { 1.0f32 } else { 0.0f32 }; 

    match axis {
        0 => vek::Vec3::<f32>::new(normal, u, v),
        1 => vek::Vec3::<f32>::new(u, normal, v),
        2 => vek::Vec3::<f32>::new(u, v, normal),
        _ => panic!()
    }
}

/*
// face in [0..5]
fn get_face_normal(face: u32) {
    let axis = face % 3;
    let positive = face / 3;
    vek::Vec3::unit_x()
}
*/

const FACE_PAIRS: [(usize, usize); 15] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 5),
];

fn face_pair_index(face_1: usize, face_2: usize) -> usize {
    match (face_1, face_2) {
        (0, 1) => 0,
        (0, 2) => 1,
        (0, 3) => 2,
        (0, 4) => 3,
        (0, 5) => 4,
        (1, 2) => 5,
        (1, 3) => 6,
        (1, 4) => 7,
        (1, 5) => 8,
        (2, 3) => 9,
        (2, 4) => 10,
        (2, 5) => 11,
        (3, 4) => 12,
        (3, 5) => 13,
        (4, 5) => 14,
        _ => panic!()
    }
}


// face pair index is between 0 and 15, so we need 4 bits
const NUM_BITS_FACE_PAIR: usize = 4;

// each face will have 16*16=16 segments in total, so we need 8 bits
const NUM_BITS_SEGMENTS: usize = 8;

// segment iter size in one axis
const SUBDIVISIONS: usize = 16;
const TOTAL_SUBDIVIONS: usize = SUBDIVISIONS*SUBDIVISIONS;

// pre-allocate the buffer
const COUNT: usize = 1 << (NUM_BITS_FACE_PAIR+NUM_BITS_SEGMENTS+NUM_BITS_SEGMENTS); 

// add extra checks whenever we move in the space to make it more sensitive
const WITH_RADIUS: bool = false;

// strength applied to the [start, end] positions before we do DDA
const NORMAL_INSET_STRENGTH: f32 = 0.008f32;

// enables the bits for a specific location
fn enable_bitmask(position: vek::Vec3<i32>, mask: &mut u64) {
    let min = vek::Vec3::<i32>::broadcast(0);
    let max = vek::Vec3::<i32>::broadcast(4);

    if WITH_RADIUS {
        for x in -1..2 {
            for y in -1..2 {
                for z in -1..2 {
                    let offset = vek::Vec3::<i32>::new(x, y, z);
                    let offsetted = offset + position;

                    if offsetted.partial_cmpge(&min).reduce_and() && offsetted.partial_cmplt(&max).reduce_and() {
                        let local = offsetted.map(|x| x % 4);
                        let index = local.z + local.y * 4 + local.x * 4 * 4;
                        assert!(index >= 0 && index < 64);
                        *mask |= 1 << index;
                    }
                }    
            }        
        }
    } else {
        if position.partial_cmpge(&min).reduce_and() && position.partial_cmplt(&max).reduce_and() {
            let local = position.map(|x| x % 4);
            let index = local.z + local.y * 4 + local.x * 4 * 4;
            assert!(index >= 0 && index < 64);
            *mask |= 1u64 << (index as u64);
        }
    }
}

// input parameters are in world space
// returns the bitmask of the visited nodes
pub fn dda_bake_mask(
    face: usize,
    ray_pos: vek::Vec3<f32>,
    ray_dir: vek::Vec3<f32>,
) -> u64 {
    if face == 3 {
        //log::debug!("ray_pos: {ray_pos}, ray_dir: {ray_dir}");

    }
    let modified_ray_pos = ray_pos * 4.0f32;
    let mut floored_pos = modified_ray_pos.floor().as_::<i32>();
    let inv_dir = ray_dir.map(|x| x.abs().max(1e-10)).recip();
    let dir_sign = ray_dir.map(|x| x.signum());
    let mut side_dist = ((dir_sign * (floored_pos.as_::<f32>() - modified_ray_pos) + dir_sign * 0.5 + 0.5) * inv_dir);
    let mut mask = 0u64;

    let min = vek::Vec3::<i32>::broadcast(0);
    let max = vek::Vec3::<i32>::broadcast(4);

    let mut within_volume = false;

    for i in 0..3 {
        if (face == 3) {
            //log::debug!("iter: {i}, pos: {floored_pos}, side dist: {side_dist}");
        }

        enable_bitmask(floored_pos, &mut mask);
        if floored_pos.partial_cmpge(&min).reduce_and() && floored_pos.partial_cmplt(&max).reduce_and() {
            within_volume = true;
        }

        if (floored_pos.partial_cmplt(&min).reduce_or() || floored_pos.partial_cmpge(&max).reduce_or()) && within_volume {
            //log::debug!("break at {i}!");
            break;
        }

        let eqs = vek::Vec3::broadcast(side_dist.reduce_partial_min()).partial_cmpeq(&side_dist);
        let eqs = eqs.map(|x| if x { 1.0f32 } else { 0.0f32 });
        side_dist += eqs * inv_dir;
        floored_pos += (eqs * dir_sign).as_::<i32>();
    }

    return mask;
}

// bakes the computation masks for two faces
// symmetric
// face pair index is the highest 4 bits
// face 1 segment index is after that
// face 2 segment index is after that
pub fn bake_intersection(
    face_1: usize,
    face_2: usize,
    masks: &mut Vec<u64>,
) {
    // compute the face pair index 
    let face_pair_index = face_pair_index(face_1, face_2);
    assert!(face_pair_index < 15);
    assert!(face_2 > face_1);

    // loop over all the segments in the first face
    for x_1 in 0..SUBDIVISIONS {
        for y_1 in 0..SUBDIVISIONS {
            let face_1_segment_index = x_1 + y_1 * SUBDIVISIONS;
            assert!(face_1_segment_index < TOTAL_SUBDIVIONS);

            let half = (1.0f32 / SUBDIVISIONS as f32) * 0.5f32;
            let uv_1 = vek::Vec2::new(x_1, y_1).map(|i| i as f32 / SUBDIVISIONS as f32 + half);
            let start_position = unflatten_uv_offset(face_1, uv_1.x, uv_1.y) + FACE_NORMALS[face_1 as usize] * NORMAL_INSET_STRENGTH;

            // loop over all the segments in the second face
            for x_2 in 0..SUBDIVISIONS {
                for y_2 in 0..SUBDIVISIONS {
                    let face_2_segment_index = x_2 + y_2 * SUBDIVISIONS;
                    assert!(face_2_segment_index < TOTAL_SUBDIVIONS);

                    let uv_2 = vek::Vec2::new(x_2, y_2).map(|i| i as f32 / SUBDIVISIONS as f32 + half);
                    let end_position = unflatten_uv_offset(face_2, uv_2.x, uv_2.y) + FACE_NORMALS[face_2 as usize] * NORMAL_INSET_STRENGTH;

                    // create ray position and direction to look at the face
                    //log::debug!("start: {start_position}, end: {end_position}");
                    let ray_dir = (end_position - start_position).normalized();
                    let ray_pos = start_position;

                    // compute the index for this specific combination
                    let index = face_pair_index << (NUM_BITS_SEGMENTS + NUM_BITS_SEGMENTS) | face_1_segment_index << (NUM_BITS_SEGMENTS) | face_2_segment_index;
                    assert!(index < COUNT);
                    
                    // our index *must* be unique
                    assert_eq!(masks[index], u64::MIN);
                    
                    // get the mask and add it to our list
                    let mut mask = dda_bake_mask(face_2, ray_pos, ray_dir);

                    /*
                    // Due to floating point precision, it is NOT symmetric! Bruh!
                    let ray_dir2 = (start_position - end_position).normalized();
                    let ray_pos2 = end_position;
                    mask |= dda_bake_mask(face_2, ray_pos2, ray_dir2);
                    */

                    if (face_1 == 0 && face_2 == 3) {
                        //let mask2 = dda_bake_mask(face_2, ray_pos2, ray_dir2);
                        //assert_eq!(mask, mask2);
                    }
                    
                    // we *must* have hit something...
                    //assert_ne!(mask, 0);

                    // test...
                    /*
                    dda_test(0x0, mask, ray_pos, ray_dir);
                    dda_test(0xFF, mask, ray_pos, ray_dir);
                    dda_test(0xAF_03_20_40_EF_01_45_15, mask, ray_pos, ray_dir);
                    dda_test(0x42_AF_03_20_40_EF_01_45, mask, ray_pos, ray_dir);
                    dda_test(0x13_42_AF_03_20_40_EF_01, mask, ray_pos, ray_dir);
                    */

                    masks[index] = mask;
                }
            }
        }
    }
}

// does all the intersection tests between two faces
pub fn bake_all() -> Vec<u64> {
    log::info!("baking micro-voxel buffer...");
    let mut masks =  (0..COUNT).into_iter().map(|_| 0u64).collect::<Vec<_>>();

    for (face_1, face_2) in FACE_PAIRS {
        log::info!("baking intersection (f1: {face_1}, f2: {face_2})");
        bake_intersection(face_1, face_2, &mut masks);
    }

    masks
}

pub unsafe fn create_dda_precomputed_buffer(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    allocator: &mut Allocator,
    binder: &Option<ash::ext::debug_utils::Device>,
) -> crate::buffer::Buffer {
    let baked = bake_all();
    let data = bytemuck::cast_slice::<u64, u8>(&baked);
    let buffer = crate::buffer::create_buffer(device, allocator, COUNT * size_of::<u64>(), binder, "baked dda buffer");
    crate::buffer::write_to_buffer(device, pool, queue, buffer.buffer, allocator, data);
    
    return buffer;
}