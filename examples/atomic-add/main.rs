use kiyo::app::app::{App, AppConfig};
use kiyo::app::draw_orch::{AtomicClearConfig, ClearConfig, DispatchConfig, DrawConfig, ImageConfig, Pass};
use kiyo::app::audio_orch::{AudioConfig};

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
                clear: ClearConfig::Color(0.0, 0.0, 0.0)
            },
        ]),
        atomic_image: AtomicClearConfig::Color(0, 0, 0),
        passes: Vec::from([
            Pass {
                shader: "examples/atomic-add/shaders/colors.comp".to_string(),
                dispatches: DispatchConfig::Count(1, 1, 1),
                input_resources: Vec::from([]),
                output_resources: Vec::from([ 0 ]),
            },
            Pass {
                shader: "examples/atomic-add/shaders/readback.comp".to_string(),
                dispatches: DispatchConfig::FullScreen,
                input_resources: Vec::from([]),
                output_resources: Vec::from([ 0 ]),
            },
        ])
    };

    App::run(app_config, config, AudioConfig::None);
}
