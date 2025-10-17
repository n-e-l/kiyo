use kiyo::app::App;
use kiyo::app::app::AppConfig;
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
        images: Vec::from([
            ImageConfig {
                clear: ClearConfig::None
            },
            ImageConfig {
                clear: ClearConfig::None
            },
        ]),
        atomic_image: AtomicClearConfig::None,
        passes: Vec::from([
            Pass {
                shader: "examples/feedback/shaders/setup.comp".to_string(),
                dispatches: DispatchConfig::FullScreen,
                input_resources: Vec::from([]),
                output_resources: Vec::from([ 0 ]),
            },
            Pass {
                shader: "examples/feedback/shaders/blur.comp".to_string(),
                dispatches: DispatchConfig::FullScreen,
                input_resources: Vec::from([ 0 ]),
                output_resources: Vec::from([ 1 ]),
            },
            Pass {
                shader: "examples/feedback/shaders/sharpen.comp".to_string(),
                dispatches: DispatchConfig::FullScreen,
                input_resources: Vec::from([ 1 ]),
                output_resources: Vec::from([ 0 ]),
            },
        ])
    };

    App::run(app_config, config, AudioConfig::None);
}
