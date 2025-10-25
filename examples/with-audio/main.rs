use kiyo::app::app::{App, AppConfig};
use kiyo::app::draw_orch::{AtomicClearConfig, ClearConfig, DispatchConfig, DrawConfig, ImageConfig, Pass};
use kiyo::app::audio_orch::{AudioConfig};

fn main() {
    const USE_FILE:bool = true;

    let app_config = AppConfig {
        width: 1000,
        height: 1000,
        fullscreen: false,
        vsync: true,
        log_fps: false,
    };

    let config = DrawConfig {
        width: 1920,
        height: 1080,
        images: Vec::from([
            ImageConfig {
                clear: ClearConfig::Color(0.0, 0.0, 0.0)
            },
        ]),
        atomic_image: AtomicClearConfig::None,
        passes: Vec::from([
            Pass {
                shader: "examples/with-audio/shaders/colors.comp".to_string(),
                dispatches: DispatchConfig::FullScreen,
                input_resources: Vec::from([]),
                output_resources: Vec::from([ 0 ]),
            },
        ])
    };

    // t: time in seconds
    fn audio_shader(t:f32) -> (f32, f32) {
        let tau = 2.0 * std::f32::consts::PI;
        let n = f32::sin(tau * 440.0 * t);
        let m = n*f32::powf(1.0-t,3.0);
        let a = (f32::sin(t*tau)/2.0-0.5)*m;
        let b = (f32::sin(t*tau + tau*0.5)/2.0-0.5)*m;

        (a, b)
    }
    App::run(app_config, config, if USE_FILE 
        {AudioConfig::AudioFile("examples/with-audio/sample.wav".to_string())} else 
        {AudioConfig::Program(audio_shader)}
    );
}
