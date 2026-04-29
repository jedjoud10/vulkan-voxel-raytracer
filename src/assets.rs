use std::{
    fs::File,
    io::{BufReader, Read},
    mem::ManuallyDrop,
    path::Path,
};

/*
pub fn damn<P: AsRef<Path>>(path: P) -> Vec<u8> {
    let file = File::open(path).unwrap();
    let mut bytes = Vec::<u8>::new();
    BufReader::new(file).read_to_end(&mut bytes).unwrap();
    return bytes;
}
*/

#[macro_export]
macro_rules! asset {
    ($file:expr, $assets:expr) => {{
        let bytes = include_bytes!(env!($file));
        let (_, words, _) = bytemuck::pod_align_to::<u8, u32>(bytes);
        $assets.insert($file, words);
        log::info!("loading embedded asset '{}' from compile time...", $file);
        /*
        cfg_if::cfg_if! {
            if #[cfg(debug_assertions)] {
                {
                    $assets.insert($file, convert(damn(env!($file))));
                    log::info!("loading asset '{}' dynamically at runtime...", $file);
                }
            } else {
                let bytes = include_bytes!(env!($file));
                $assets.insert($file, convert(bytes.to_vec()));
                log::info!("loading embedded asset '{}' from compile time...", $file);
            }
        }
        */
    }};
}
