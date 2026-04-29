use std::time::{Duration, Instant};

pub struct Benchmark {
    starting_frame: u64,
    starting_instant: Instant,
    timings: Vec<f64>,
}

pub struct Statistics {
    benchmark: Option<Benchmark>,
    delta_ms_buffer: [f64; 8],
    benchmark_duration: Duration,
}

impl Default for Statistics {
    fn default() -> Self {
        Self {
            benchmark: Default::default(),
            delta_ms_buffer: Default::default(),
            benchmark_duration: Duration::from_secs(2),
        }
    }
}

impl Statistics {
    pub fn push_query_timings(&mut self, delta_in_ms: f64) {
        self.delta_ms_buffer.rotate_right(1);
        self.delta_ms_buffer[0] = delta_in_ms;
    }

    pub fn end_of_frame(&mut self, frame: u64) {
        let avg = self.get_average_in_ms();
        if let Some(benchmark) = self.benchmark.as_mut() && frame > (benchmark.starting_frame + self.delta_ms_buffer.len() as u64)  {
            let elapsed = Instant::now() - benchmark.starting_instant;
            benchmark.timings.push(avg);
        
            if elapsed > self.benchmark_duration {
                self.end_benchmarking();
            }
        }
    }

    pub fn start_benchmarking(&mut self, frame: u64) {
        log::info!("start benchmarking");
        self.benchmark = Some(Benchmark {
            starting_frame: frame,
            starting_instant: Instant::now(),
            timings: Default::default()
        });
    }

    pub fn end_benchmarking(&mut self) {
        log::info!("end benchmarking");

        let benchmark = self.benchmark.as_ref().unwrap();

        let n = benchmark.timings.len();
        let avg = benchmark.timings.iter().sum::<f64>() / (n as f64);
        let stddev = (benchmark.timings.iter().map(|x| (x - avg).powf(2.0f64)).sum::<f64>()).sqrt() / (n as f64);
        
        log::info!("Sample Count: {n}, Avg: {avg}ms, StdDev: {stddev}");

        self.benchmark = None;
    }

    pub fn get_average_in_ms(&self) -> f64 {
        return self.delta_ms_buffer.iter().sum::<f64>() / self.delta_ms_buffer.len() as f64;
    }
}