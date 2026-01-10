use std::{
    env,
    fs::File,
    io::Write,
    path::PathBuf,
};

use slang::Downcast;

fn load_module(session: &mut slang::Session, file_name: &str) {
    let module = session.load_module(&format!("{file_name}.slang")).unwrap();

    if module.entry_point_count() == 0 {
        return;
    }

    let mut component_types = Vec::<slang::ComponentType>::new();
    component_types.push(module.downcast().clone());
    for entry_point in module.entry_points() {
        component_types.push(entry_point.downcast().clone());
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
    println!("cargo:warning=Compiled! {length} bytes, saved to {path_str}");
}

// TODO: optimize. this will re-compile all shaders, even if only one of them was modified
fn main() {
    println!("cargo:rerun-if-changed=shaders");
    let global_session = slang::GlobalSession::new().unwrap();
    let search_path = std::ffi::CString::new("shaders").unwrap();

    let session_options = slang::CompilerOptions::default()
        .optimization(slang::OptimizationLevel::Default)
        .vulkan_use_entry_point_name(true)
        .matrix_layout_row(true);

    
    let target_desc = slang::TargetDesc::default().format(slang::CompileTarget::Spirv);
    let targets = [target_desc];
    let search_paths = [search_path.as_ptr()];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let mut session = global_session.create_session(&session_desc).unwrap();

    let mut dir_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    dir_path.push("shaders");
    let dir = std::fs::read_dir(dir_path).unwrap();

    for entry in dir.filter_map(|x| x.ok()) {
        let file_name = entry.file_name().into_string().unwrap();
        let file_name = file_name.split(".").next().unwrap();
        load_module(&mut session, file_name);
    }
}
