use std::{
    env,
    fs::{self, DirEntry, File},
    io::Write,
    path::{Path, PathBuf},
};

use shader_slang::*;

fn load_module(session: &Session, file_name: &str) {
    let module: Module = session.load_module(&format!("{file_name}.slang")).unwrap();

    if module.entry_point_count() == 0 {
        return;
    }

    let mut component_types = Vec::<ComponentType>::new();
    component_types.push(ComponentType::from(module.clone()));
    for entry_point in module.entry_points() {
        component_types.push(ComponentType::from(entry_point));
    }

    let program = session.create_composite_component_type(&component_types).unwrap();
    let linked_program = program.link().unwrap();

    // ERROR: for some reason `target_code` does not return Err, even when there is no valid target code
    // it just gives an undefined blob. program crashes when you try to `as_slice` it.
    // TODO: report as issue?
    let shader_bytecode = linked_program.target_code(0).unwrap();
    let raw = shader_bytecode.as_slice();
    let length = raw.len();

    let out_dir = env::var("OUT_DIR").unwrap();
    let mut path = PathBuf::from(out_dir);
    path.push(format!("{file_name}.spv"));

    let mut file = File::create(&path).unwrap();
    file.write(shader_bytecode.as_slice()).unwrap();

    let path_str = path.to_str().unwrap();
    println!("cargo:rustc-env={file_name}.spv={path_str}");
    println!("cargo:warning={file_name} compiled to {length} bytes");
}

// https://doc.rust-lang.org/nightly/std/fs/fn.read_dir.html#examples
fn visit_dirs(dir: &Path, list: &mut Vec<DirEntry>) {
    if dir.is_dir() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if entry.path().is_dir() {
                visit_dirs(&entry.path(), list);
            } else {
                list.push(entry);
            }
        }
    }
}

// TODO: optimize. this will re-compile all shaders, even if only one of them was modified
fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let global_session = GlobalSession::new().unwrap();

    let session_options = CompilerOptions::default()
        .optimization(OptimizationLevel::Maximal)
        .debug_information(DebugInfoLevel::None)
        .obfuscate(false)
        .no_mangle(false)
        .vulkan_use_entry_point_name(true)
        .matrix_layout_row(true);

    
    let target_desc = TargetDesc::default().format(CompileTarget::Spirv);
    let targets = [target_desc];
    let search_paths = [c"shaders".as_ptr(), c"shaders/noises".as_ptr()];

    let session_desc = SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = global_session.create_session(&session_desc).unwrap();

    let mut dir_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    dir_path.push("shaders");
    let mut entries = Vec::<DirEntry>::new();
    visit_dirs(&dir_path, &mut entries);

    for entry in entries {
        let file_name = entry.file_name().into_string().unwrap();
        let file_name = file_name.split(".").next().unwrap();
        load_module(&session, file_name);
    }
}
