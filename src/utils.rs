pub fn pseudo_random(seed: u32) -> u32 {
    let mut value = seed;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    value = value.wrapping_mul(0x322adf);
    value ^= value >> 11;
    value = value.wrapping_add(0x9e3779b9);
    value
}

pub fn is_set(bitmask: u64, index: u32) -> bool {
    ((bitmask >> index) & 1) == 1
}

pub fn set_bit(mask: &mut u64, index: u32, set: bool) {
    if set {
        *mask |= 1 << index;
    } else{
        *mask &= !(1 << index);
    }
}

pub const OFFSETS: [vek::Vec3::<i32>; 6] = [
    vek::Vec3::new(-1, 0, 0),
    vek::Vec3::new(1, 0, 0),
    vek::Vec3::new(0, -1, 0),
    vek::Vec3::new(0, 1, 0),
    vek::Vec3::new(0, 0, -1),
    vek::Vec3::new(0, 0, 1),
];


#[macro_export]
macro_rules! asset {
    ($file:expr, $assets:expr) => {{
        let bytes = include_bytes_aligned::include_bytes_aligned!(4, env!($file));
        let words = bytemuck::cast_slice::<u8, u32>(bytes);
        $assets.insert($file, words);
        log::info!("loading embedded asset '{}' from compile time...", $file);
    }};
}
