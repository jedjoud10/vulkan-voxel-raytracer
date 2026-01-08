use crate::input::{Axis, Input, MouseAxis};
use vek::Clamp;
use winit::keyboard::KeyCode;

#[derive(Default)]
pub struct Movement {
    pub position: vek::Vec3<f32>,
    pub rotation: vek::Quaternion<f32>,
    pub proj_matrix: vek::Mat4<f32>,
    pub view_matrix: vek::Mat4<f32>,

    pub summed_mouse: vek::Vec2<f32>,
    pub local_velocity: vek::Vec2<f32>,
    pub velocity: vek::Vec3<f32>,
    pub boost: f32,
}

impl Movement {
    pub fn new() -> Self {
        Self {
            position: vek::Vec3::new(crate::voxel::SIZE as f32 / 2f32, 20f32, crate::voxel::SIZE as f32 / 2f32),
            ..Default::default()
        }
    }

    pub fn update(&mut self, input: &Input, ratio: f32, delta: f32) {
        self.local_velocity = vek::Vec2::<f32>::zero();
        let speed = if input.get_button(KeyCode::ShiftLeft).held() {
            2f32 + 2f32.powf(self.boost)
        } else if input.get_button(KeyCode::ControlLeft).held() {
            0.25f32
        } else {
            1.0f32
        };

        if input.get_button(KeyCode::KeyW).held() {
            self.local_velocity.y = 1f32;
        } else if input.get_button(KeyCode::KeyS).held() {
            self.local_velocity.y = -1f32;
        }

        if input.get_button(KeyCode::KeyA).held() {
            self.local_velocity.x = 1f32;
        } else if input.get_button(KeyCode::KeyD).held() {
            self.local_velocity.x = -1f32;
        }

        self.boost += input.get_axis(Axis::Mouse(MouseAxis::ScrollDelta));
        self.boost = self.boost.clamp(0.0, 5.0);
        let sens = 1.0f32;
        let summed_mouse_target = vek::Vec2::new(
            input.get_axis(Axis::Mouse(MouseAxis::PositionX)) * 0.003 * sens,
            input.get_axis(Axis::Mouse(MouseAxis::PositionY)) * -0.003 * sens,
        );
        self.summed_mouse = vek::Vec2::lerp(
            self.summed_mouse,
            summed_mouse_target,
            (40f32 * delta).clamped01(),
        );
        self.rotation = vek::Quaternion::rotation_y(self.summed_mouse.x)
            * vek::Quaternion::rotation_x(self.summed_mouse.y);

        let uhh = 1f32 / ratio;
        // TODO: fix the weird radian fov?
        self.proj_matrix =
            vek::Mat4::<f32>::perspective_rh_no(80.0f32.to_radians(), uhh, 0.001f32, 1000f32);
        let rot = vek::Mat4::from(self.rotation);

        let forward = rot.mul_direction(-vek::Vec3::unit_z());
        let right = rot.mul_direction(vek::Vec3::unit_x());
        let up = rot.mul_direction(vek::Vec3::unit_y());
        self.view_matrix = vek::Mat4::look_at_rh(self.position, forward + self.position, up);

        let velocity = forward * self.local_velocity.y + right * self.local_velocity.x;
        self.velocity = vek::Vec3::lerp(
            self.velocity,
            velocity * 3.0f32 * speed,
            (40f32 * delta).clamped01(),
        );

        self.position += self.velocity * delta;
    }
}

pub fn horizontal_to_vertical(hfov: f32, ratio: f32) -> f32 {
    2.0 * ((hfov.to_radians() / 2.0).tan() * (1.0 / (ratio))).atan()
}
