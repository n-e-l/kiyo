use std::fs::File;
use std::path::Path;
use std::io::BufWriter;
use png::SrgbRenderingIntent;

fn linear_to_srgb(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}


pub fn write_png_image(in_data: &[u8], width: u32, height: u32, path: &str ) {
    let path = Path::new(path);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut srgb_data = vec![0u8; in_data.len()];
    for i in (0..in_data.len()).step_by(4) {
        let r_linear = in_data[i] as f32 / 255.0;
        let g_linear = in_data[i + 1] as f32 / 255.0;
        let b_linear = in_data[i + 2] as f32 / 255.0;

        srgb_data[i] = (linear_to_srgb(r_linear) * 255.0) as u8;
        srgb_data[i + 1] = (linear_to_srgb(g_linear) * 255.0) as u8;
        srgb_data[i + 2] = (linear_to_srgb(b_linear) * 255.0) as u8;
        srgb_data[i + 3] = in_data[i + 3]; // Alpha unchanged
    }

    let mut encoder = png::Encoder::new(w, width, height );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));
    encoder.set_source_srgb(SrgbRenderingIntent::Perceptual);

    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data( &srgb_data ).unwrap(); // Save
}