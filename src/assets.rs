#[macro_export]
macro_rules! asset {
    ($file:expr, $assets:expr) => {{
        let bytes = include_bytes_aligned::include_bytes_aligned!(4, env!($file));
        let words = bytemuck::cast_slice::<u8, u32>(bytes);
        $assets.insert($file, words);
        log::info!("loading embedded asset '{}' from compile time...", $file);
    }};
}
