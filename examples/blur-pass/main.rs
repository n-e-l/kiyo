use kiyo::app::app::{App, AppConfig};
use kiyo::app::audio_orch::AudioConfig;
use kiyo::app::draw_orch::{AtomicClearConfig, ClearConfig, DispatchConfig, DrawConfig, ImageConfig, Pass};

fn main() {

    let app_config = AppConfig {
        width: 1000,
        height: 1000,
        vsync: true,
        log_fps: false,
        fullscreen: false,
    };

    let config = DrawConfig {
        width: 1920,
        height: 1080,
        images: Vec::from([
            ImageConfig {
                clear: ClearConfig::Color(0.0, 0.0, 0.0)
            },
            ImageConfig {
                clear: ClearConfig::Color(0.0, 0.0, 0.0)
            },
        ]),
        atomic_image: AtomicClearConfig::None,
        passes: Vec::from([
            Pass {
                shader: "examples/blur-pass/shaders/screen_shader.comp".to_string(),
                dispatches: DispatchConfig::FullScreen,
                input_resources: Vec::from([]),
                output_resources: Vec::from([ 0 ]),
            },
            Pass {
                shader: "examples/blur-pass/shaders/blur.comp".to_string(),
                dispatches: DispatchConfig::FullScreen,
                input_resources: Vec::from([ 0 ]),
                output_resources: Vec::from([ 1 ]),
            }
        ])
    };

    App::run(app_config, config, AudioConfig::None);
}
