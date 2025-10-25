use kiyo::app::app::{App, AppConfig};
use kiyo::app::audio_orch::AudioConfig;
use kiyo::app::draw_orch::{AtomicClearConfig, ClearConfig, DrawConfig, ImageConfig};

fn main() {

    let app_config = AppConfig {
        width: 1000,
        height: 1000,
        vsync: true,
        log_fps: false,
        fullscreen: false,
    };

    // Display a single image cleared to yellow
    let config = DrawConfig {
        width: 1920,
        height: 1080,
        images: Vec::from([
            ImageConfig {
                clear: ClearConfig::Color(1.0, 1.0, 0.0)
            },
        ]),
        atomic_image: AtomicClearConfig::None,
        passes: Vec::from([
        ])
    };

    App::run(app_config, config, AudioConfig::None);
}
