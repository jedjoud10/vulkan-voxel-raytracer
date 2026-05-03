#![allow(unsafe_op_in_unsafe_fn)]

mod debug;
mod device;
mod input;
mod instance;
mod movement;
mod physical_device;
mod pipeline;
mod swapchain;
mod voxel;
mod ticker;
mod buffer;
mod rays;
mod statistics;
mod utils;
mod renderer;
mod skybox;
mod others;
mod constant_descriptor_sets;
mod per_frame_data;

use ash;
use ash::vk;
use clap::Parser;
use gpu_allocator::vulkan::Allocation;
use input::Button;
use input::Input;
use movement::Movement;
use winit::event::MouseButton;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::KeyCode;
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::{Window, WindowId};
use statistics::Statistics;
use renderer::InternalApp;


#[derive(clap::Parser, Debug)]
#[command(about = "Vulkan DDA Voxel Raytracer", long_about = None)]
struct Args {
    /// Factor to use to decrease the screen resolution
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=4))]
    downscale_factor: u32,

    /// Number of shadow samples to use. Set to 0 to disable shadows completely. Set to 1 to use hard-shadows.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(0..=16))]
    shadow_samples: u32,

    /// Maximum number of rays to trace iteratively for reflections / refractions
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=8))]
    max_ray_iterations: u32,

    /// Whether or not to use round spherical normals
    #[arg(long, default_value_t = false)]
    round_normals: bool,

    /// Whether or not to use ray traced ambient occlusion
    #[arg(long, default_value_t = false)]
    ambient_occlusion: bool,

    /// Fun setting to make all mirror reflections wavey lolol
    #[arg(long, default_value_t = true)]
    wavy_reflections: bool,

    /// Setting to make all shadows pixelated
    #[arg(long, default_value_t = false)]
    pixelated_shadows: bool,

    /// Setting to start in fullscreen from the start. This can be toggled in-game using F5
    #[arg(long, default_value_t = false)]
    fullscreen: bool,

    /// Group size exponent used for the main ray-tracing shader. Ex a value of 3 means using 2^3=8, a group size of 8x8
    /// TODO: find better name for this
    #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=5))]
    group_size_exp: u32,
}

struct App {
    internal: Option<InternalApp>,
    args: Option<Args>,
    start: Instant,
    last: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        unsafe {
            self.internal = Some(InternalApp::new(event_loop, self.args.take().unwrap()));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => unsafe {
                event_loop.exit();
                self.internal.take().unwrap().destroy();
            },
            WindowEvent::RedrawRequested => unsafe {
                let inner = self.internal.as_mut().unwrap();
                let new = Instant::now();
                let elapsed = (new - self.start).as_secs_f32();
                let delta = (new - self.last).as_secs_f32();

                if let ControlFlow::Break(_) = inner.pre_render(delta) {
                    event_loop.exit();
                    self.internal.take().unwrap().destroy();
                    return;
                }

                inner.window.request_redraw();
                inner.render(delta, elapsed);
                self.last = new;
                input::update(&mut inner.input);
            },
            WindowEvent::Resized(new) => unsafe {
                let inner = self.internal.as_mut().unwrap();
                inner.resize(new.width, new.height);
            },

            // This is horrid...
            _ => {
                let inner = self.internal.as_mut().unwrap();
                input::window_event(&mut inner.input, &event);
            }
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let inner = self.internal.as_mut().unwrap();
        input::device_event(&mut inner.input, &event);
    }
}

pub fn main() {
    let args = Args::parse();
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Debug)
        .init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App {
        start: Instant::now(),
        last: Instant::now(),
        internal: None,
        args: Some(args),
    };
    event_loop.run_app(&mut app).unwrap();
}
