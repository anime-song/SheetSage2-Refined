//! Decode and resample an audio file as the engine does and write it as raw float32 (for comparing with
//! `transcribe.load_waveform`): `cargo run --release --example decode -- IN OUT.f32 [--no-resample]`

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (wave, rate) = sheetsage2_cpu::audio::decode(std::path::Path::new(&args[1])).expect("decode");
    let wave = if args.iter().any(|a| a == "--no-resample") {
        wave
    } else {
        sheetsage2_cpu::audio::resample(&wave, rate as usize, sheetsage2_cpu::encoder::SAMPLE_RATE)
    };
    let bytes: Vec<u8> = wave.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[2], bytes).expect("write");
    eprintln!("{} samples at {rate} Hz", wave.len());
}
