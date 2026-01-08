pub struct Ticker {
    pub ticks_per_second: f32,
    pub accumulator: f32,
    pub count: u32,
}

impl Ticker {
    pub fn update(&mut self, delta: f32) -> bool {
        self.accumulator += delta;

        // For now we assume we can execute at most one tick per frame
        // TODO: implement multiple ticks per frame, but this might causes some synhronization difficulties since
        // we do the update on the GPU
        if self.accumulator > (1f32 / self.ticks_per_second) {
            self.accumulator = 0f32;
            self.count += 1;
            return true;
        }

        return false;
    }
}