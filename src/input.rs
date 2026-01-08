/// Mouse axis that we can bind to an axis mapping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseAxis {
    /// X position of the mouse
    PositionX,

    /// Y position of the mouse
    PositionY,

    /// Current scroll value (integral of ScrollDelta)
    Scroll,

    /// Derivative of the X position of the mouse
    DeltaX,

    /// Derivative of the Y position of the mouse
    DeltaY,

    /// How much the scroll wheel scrolled
    ScrollDelta,
}

/// An axis can be mapped to a specific binding to be able to fetch it using a user defined name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Axis {
    /// Mouse axii.
    Mouse(MouseAxis),
}
impl From<MouseAxis> for Axis {
    fn from(value: MouseAxis) -> Self {
        Axis::Mouse(value)
    }
}

use std::collections::{hash_map::Entry, HashMap};

use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    keyboard::PhysicalKey,
};

/// The current state of any key / button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonState {
    /// The button just got pressed this frame.
    Pressed,

    /// The button was pressed/held last frame, but not this frame.
    Released,

    /// The button was kept pressed from last frame till this frame.
    Held,

    /// The button was not touched this frame nor last frame.
    None,
}

impl From<ElementState> for ButtonState {
    fn from(state: ElementState) -> Self {
        match state {
            ElementState::Pressed => Self::Pressed,
            ElementState::Released => Self::Released,
        }
    }
}

impl ButtonState {
    /// This checks if the state is equal to State::Pressed.
    pub fn pressed(&self) -> bool {
        matches!(self, ButtonState::Pressed)
    }

    /// This checks if the state is equal to State::Released.
    pub fn released(&self) -> bool {
        matches!(self, ButtonState::Released)
    }

    /// This checks if the State is equal to State::Held.
    pub fn held(&self) -> bool {
        matches!(self, ButtonState::Held)
    }
}

/// Keyboard button / key
pub type KeyboardButton = winit::keyboard::KeyCode;

/// Mouse button / key
pub type MouseButton = winit::event::MouseButton;

/// A button that might come from the keyboard, mouse, or a gamepad
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
#[repr(u32)]
pub enum Button {
    /// Any sort of keyboard button
    Keyboard(KeyboardButton),

    /// Mouse buttons that we can press
    Mouse(MouseButton),
}

impl From<KeyboardButton> for Button {
    fn from(value: KeyboardButton) -> Self {
        Button::Keyboard(value)
    }
}

impl From<MouseButton> for Button {
    fn from(value: MouseButton) -> Self {
        Button::Mouse(value)
    }
}

/// Trait implemented for structs that allow us to fetch the key state from the main input handler.
/// Allows for use to pass `&str` and `Button` interchangeably into methods that require fetching button state.
pub trait InputButtonId {
    /// Get the button state using `self` as an identifier
    fn get(self, input: &Input) -> ButtonState;
}

impl<T: Into<Button>> InputButtonId for T {
    fn get(self, input: &Input) -> ButtonState {
        let converted = self.into();
        input
            .keys
            .get(&converted)
            .cloned()
            .unwrap_or(ButtonState::None)
    }
}

impl InputButtonId for &'static str {
    fn get(self, input: &Input) -> ButtonState {
        input
            .bindings
            .key_bindings
            .get(self)
            .map(|key| Button::get(*key, input))
            .unwrap_or(ButtonState::None)
    }
}

/// Trait implemented for structs that allow us to fetch the axis state from the main input handler.
/// Allows for use to pass `&str` and `Axis` interchangeably into methods that require fetching button state.
pub trait InputAxisId {
    /// Get the input state using `self` as an identifier
    fn get(self, input: &Input) -> f32;
}

impl<T: Into<Axis>> InputAxisId for T {
    fn get(self, input: &Input) -> f32 {
        let converted = self.into();
        input.axii.get(&converted).cloned().unwrap_or_default()
    }
}

impl InputAxisId for &'static str {
    fn get(self, input: &Input) -> f32 {
        input
            .bindings
            .axis_bindings
            .get(self)
            .map(|axis| Axis::get(*axis, input))
            .unwrap_or_default()
    }
}

/// Main input resource responsible for keyboard / mouse / gamepad input events.
/// This resource will automatically be added into the world at startup.
#[derive(Default)]
pub struct Input {
    // Key and axis bindings
    pub(crate) bindings: InputUserBindings,

    // Key::W -> State::Pressed
    pub(crate) keys: HashMap<Button, ButtonState>,

    // Axis::MousePositionX -> 561.56
    pub(crate) axii: HashMap<Axis, f32>,
}

/// User input bindings that can be serialized / deserialized.
#[derive(Default, Clone)]
pub struct InputUserBindings {
    // "forward_key_bind" -> Key::W
    pub(crate) key_bindings: HashMap<&'static str, Button>,

    // "camera rotation" -> Axis:MousePositionX,
    pub(crate) axis_bindings: HashMap<&'static str, Axis>,
}

impl Input {
    /// Create a new button binding using a name and a unique key.
    pub fn bind_button(&mut self, name: &'static str, key: impl Into<Button>) {
        let key = key.into();
        log::debug!("Binding button/key {key:?} to '{name}'");
        self.bindings.key_bindings.insert(name, key);
    }

    /// Create a new axis binding using a name and a unique axis.
    pub fn bind_axis(&mut self, name: &'static str, axis: impl Into<Axis>) {
        let axis = axis.into();
        log::debug!("Binding axis {axis:?} to '{name}'");
        self.bindings.axis_bindings.insert(name, axis);
    }

    /// Get the state of a button mapping or a key mapping.
    pub fn get_button<B: InputButtonId>(&self, button: B) -> ButtonState {
        B::get(button, self)
    }

    /// Get the state of a unique axis or an axis mapping.
    pub fn get_axis<A: InputAxisId>(&self, axis: A) -> f32 {
        A::get(axis, self)
    }
}

// Winit window event since it seems that DeviceEvent::Key is broken on other machines
// TODO: Report bug
pub fn window_event(input: &mut Input, ev: &WindowEvent) {
    fn handle_button_input(input: &mut Input, key: Button, state: ElementState) {
        match input.keys.entry(key) {
            Entry::Occupied(mut current) => {
                // Check if the key is "down" (either pressed or held)
                let down = matches!(*current.get(), ButtonState::Pressed | ButtonState::Held);

                // If the key is pressed while it is currently down, it repeated itself, and we must ignore it
                if down ^ (state == ElementState::Pressed) {
                    current.insert(state.into());
                }
            }
            Entry::Vacant(v) => {
                v.insert(state.into());
            }
        }
    }

    match ev {
        // Handles keyboard keys
        WindowEvent::KeyboardInput {
            device_id,
            event:
                KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    logical_key,
                    text,
                    location,
                    state,
                    repeat,
                    ..
                },
            is_synthetic,
        } => {
            handle_button_input(input, Button::Keyboard(*code), *state);
        }

        // Handles mouse buttons
        WindowEvent::MouseInput { state, button, .. } => {
            handle_button_input(input, Button::Mouse(*button), *state);
        }

        _ => {}
    }
}

// Winit device event (called by handler when needed)
pub fn device_event(input: &mut Input, ev: &DeviceEvent) {
    match ev {
        // Update mouse position delta and summed  pos
        DeviceEvent::MouseMotion { delta } => {
            let delta = vek::Vec2::<f64>::from(*delta).as_::<f32>();
            input.axii.insert(Axis::Mouse(MouseAxis::DeltaX), delta.x);
            input.axii.insert(Axis::Mouse(MouseAxis::DeltaY), delta.y);
            let x = input
                .axii
                .entry(Axis::Mouse(MouseAxis::PositionX))
                .or_insert(0.0);
            *x += delta.x;
            let y = input
                .axii
                .entry(Axis::Mouse(MouseAxis::PositionY))
                .or_insert(0.0);
            *y += delta.y;
        }

        // Update mouse wheel delta and summed value
        DeviceEvent::MouseWheel { delta } => {
            let delta = match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                winit::event::MouseScrollDelta::PixelDelta(physical) => physical.x as f32,
            };

            input
                .axii
                .insert(Axis::Mouse(MouseAxis::ScrollDelta), delta);
            let scroll = input
                .axii
                .entry(Axis::Mouse(MouseAxis::Scroll))
                .or_insert(0.0);
            *scroll += delta;
        }

        _ => {}
    }
}

// Update event that will change the state of the keyboard keys (some states are sticky while others are not sticky)
// This will also read the state from gamepads using gilrs
pub fn update(input: &mut Input) {
    // Update the state of the keys/buttons
    for (_, state) in input.keys.iter_mut() {
        *state = match state {
            ButtonState::Pressed => ButtonState::Held,
            ButtonState::Released => ButtonState::None,
            ButtonState::Held => ButtonState::Held,
            ButtonState::None => ButtonState::None,
        };
    }

    // Reset the mouse scroll delta (since winit doesn't reset it for us)
    if let Some(data) = input.axii.get_mut(&Axis::Mouse(MouseAxis::ScrollDelta)) {
        *data = 0f32;
    }
}
