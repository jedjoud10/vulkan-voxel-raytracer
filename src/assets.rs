#[macro_export]
macro_rules! asset {
    ($file:expr, $assets:expr) => {{
        let bytes = include_bytes!(env!($file));
        let (_, words, _) = bytemuck::pod_align_to::<u8, u32>(bytes);
        $assets.insert($file, words);
        log::info!("loading embedded asset '{}' from compile time...", $file);
    }};
}
