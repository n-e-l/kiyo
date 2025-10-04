use std::collections::HashMap;
use std::mem::size_of;
use cen::ash::vk;
use cen::ash::vk::{BufferImageCopy, BufferUsageFlags, DeviceSize, Extent3D, ImageAspectFlags, ImageLayout, ImageSubresourceLayers, ImageUsageFlags, Offset3D};
use bytemuck::{Pod, Zeroable};
use cen::graphics::pipeline_store::{PipelineConfig, PipelineKey};
use cen::graphics::Renderer;
use cen::graphics::renderer::{RenderComponent, RenderContext};
use cen::vulkan::{Allocator, Buffer, CommandBuffer, DescriptorSetLayout, Device, Image, PipelineErr};
use glam::{UVec3};
use log::{error, info};
use crate::app::audio_orch::{AudioConfig};
use crate::app::audio_orch::AudioConfig::AudioFile;
use std::fs::File;
use std::io::BufReader;
use rodio::{Decoder, OutputStream, Sink};
use core::time::{Duration};
use std::ops::Add;
use std::process::exit;
use std::thread;
use cen::app::engine::InitContext;
use cen::app::gui::{GuiComponent, GuiSystem};
use cen::egui::{menu, Context, TopBottomPanel};
use cen::egui::epaint::ColorMode::UV;
use cen::gpu_allocator::MemoryLocation;
use crate::app::png::{write_png_image};

#[derive(Copy)]
#[derive(Clone)]
pub enum DispatchConfig
{
    Count( u32, u32, u32 ),
    FullScreen,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct PushConstants {
    pub time: f32,
    pub in_image: i32,
    pub out_image: i32,
}

pub struct Pass {
    pub shader: String,
    pub dispatches: DispatchConfig,
    pub input_resources: Vec<u32>,
    pub output_resources: Vec<u32>,
}

#[derive(Clone)]
pub enum ClearConfig {
    None,
    Color(f32,f32,f32),
}

#[derive(Clone)]
pub struct ImageConfig {
    pub clear: ClearConfig,
}

pub struct DrawConfig {
    pub passes: Vec<Pass>,
    pub images: Vec<ImageConfig>,
}

pub struct ShaderPass {
    pub dispatches: DispatchConfig,
    pub in_images: Vec<u32>,
    pub out_images: Vec<u32>,
    pub pipeline_handle: PipelineKey,
}

pub struct ImageResource {
    pub image: Image,
    pub clear: ClearConfig,
}

struct ImgExport {
    width: u32,
    height: u32,
    filename: String,
    do_export: bool,
}

/**
 *  Contains all render related structures relating to a config.
 */
pub struct DrawOrchestrator {
    draw_config: DrawConfig,
    audio_config: AudioConfig,
    audio_stream: Option<OutputStream>,
    sink: Option<Sink>,
    pub compute_descriptor_set_layout: DescriptorSetLayout,
    pub image_resources: Vec<ImageResource>,
    pub passes: Vec<ShaderPass>,
    image_export: ImgExport,
    workgroup_size: u32,
    start_time: std::time::Instant,
}

impl DrawOrchestrator {
    pub fn new(ctx: &mut InitContext, draw_config: DrawConfig, audio_config: AudioConfig) -> DrawOrchestrator {

        let image_count = draw_config.images.len() as u32;

        // Verify max referred index
        let max_reffered_image = draw_config.passes.iter()
            .map(|p| p.output_resources.iter())
            .flatten().max().unwrap_or(&0);
        if *max_reffered_image as i32 > image_count as i32 - 1 {
            error!("Image index out of bounds, provide enough image resources");
            panic!("Image index out of bounds, provide enough image resources");
        }

        // Layout
        let layout_bindings = &[
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(image_count)
                .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT)
        ];
        let compute_descriptor_set_layout = DescriptorSetLayout::new_push_descriptor(
            &ctx.device,
            layout_bindings
        );

        // Images
        let image_resources = Self::create_image_resources(&ctx.device, &mut ctx.allocator, &draw_config, ctx.swapchain_extent.width, ctx.swapchain_extent.height);

        let push_constant_ranges = Vec::from([
            vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(size_of::<PushConstants>() as u32),
        ]);

        let workgroup_size = 32;
        let mut macros: HashMap<String, String> = HashMap::new();
        macros.insert("NUM_IMAGES".to_string(), image_count.to_string());
        macros.insert("WORKGROUP_SIZE".to_string(), workgroup_size.to_string());

        // Passes
        let passes = draw_config.passes
            .iter()
            .map(|c| {
                let pipeline_handle = ctx.pipeline_store.insert(
                    PipelineConfig {
                        shader_path: c.shader.clone().into(),
                        descriptor_set_layouts: vec![compute_descriptor_set_layout.clone()],
                        push_constant_ranges: push_constant_ranges.clone(),
                        macros: macros.clone()
                    }
                )?;

                Ok(ShaderPass {
                    pipeline_handle,
                    dispatches: c.dispatches,
                    in_images: c.input_resources.clone(),
                    out_images: c.output_resources.clone(),
                })
            })
            .collect::<Result<Vec<ShaderPass>, PipelineErr>>()
            .inspect_err(|err| {
                error!("{}", err);
                exit(0);
            })
            .unwrap();

        // Audio things
        let mut audio_stream = None;
        let mut sink = None;
        if let AudioFile(file) = audio_config.clone() {
            let (stream, stream_handle) = OutputStream::try_default().unwrap();
            audio_stream = Some(stream);
            sink = Some(Sink::try_new(&stream_handle).unwrap());
            // Load a sound from a file, using a path relative to Cargo.toml
            let file = BufReader::new(File::open(file).unwrap());
            // Decode that sound file into a source
            let source = Decoder::new(file).unwrap();

            sink.as_ref().map(|sink| Sink::append(sink, source));
            sink.as_ref().map(|sink| Sink::play(sink));
        };
        Self {
            workgroup_size: 32,
            draw_config,
            audio_config,
            audio_stream,
            sink,
            compute_descriptor_set_layout,
            image_resources,
            passes,
            image_export: ImgExport {
                do_export: false,
                filename: "output".to_string(),
                width: 1920,
                height: 1080,
            },
            start_time: std::time::Instant::now(),
        }
    }

    fn export(&mut self, ctx: &mut RenderContext, width: u32, height: u32) {

        info!("Exporting...");
        let output_image = Image::new(
            &ctx.device,
            &mut ctx.allocator,
            width,
            height,
            ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_DST | ImageUsageFlags::TRANSFER_SRC
        );
        let mut buffer = Buffer::new(
            &ctx.device,
            &mut ctx.allocator,
            MemoryLocation::GpuToCpu,
            (size_of::<u8>() as u32 * 4 * width * height) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST
        );
        let image_resources = Self::create_image_resources(ctx.device, ctx.allocator, &self.draw_config, width, height);

        self.do_render(ctx, &image_resources, &output_image, ImageLayout::UNDEFINED, ImageLayout::TRANSFER_SRC_OPTIMAL);

        ctx.command_buffer.copy_image_to_buffer(
            &output_image,
            ImageLayout::TRANSFER_SRC_OPTIMAL,
            &buffer,
            &[
                BufferImageCopy::default()
                    .buffer_image_height(output_image.height())
                    .buffer_offset(0)
                    .image_extent(Extent3D::default().width(output_image.width()).height(output_image.height()).depth(1))
                    .image_offset(Offset3D::default())
                    .image_subresource(ImageSubresourceLayers::default()
                        .layer_count(1)
                        .mip_level(0)
                        .aspect_mask(ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                    )
            ]
        );

        let filename = self.image_export.filename.clone();
        ctx.run_on_finish(Box::new(move || {
            // Write png
            thread::spawn(move || {
                let memory = buffer.mapped().unwrap();
                let output_file = filename.clone().add(".png");
                write_png_image(memory.as_slice(), width, height, output_file.as_str());
                info!("Finished exporting png image to {}", output_file);
            });
        }));
    }

    /*
     * Perform a compute writing to @target_image
     */
    fn do_render(&self, ctx: &mut RenderContext, image_resources: &Vec<ImageResource>, target_image: &Image, src_layout: ImageLayout, dst_layout: ImageLayout) {

        // Clear all images with a clear config
        {
            for i in image_resources {
                ctx.command_buffer.image_barrier(
                    &i.image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::NONE,
                    vk::AccessFlags::TRANSFER_WRITE
                );

                match &i.clear {
                    ClearConfig::None => {},
                    ClearConfig::Color(r, g, b) => {
                        ctx.command_buffer.clear_color_image(
                            &i.image,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            [*r, *g, *b, 1f32]
                        );
                    }
                }

                ctx.command_buffer.image_barrier(
                    &i.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::SHADER_WRITE
                );
            }
        }

        // Compute images
        let current_time = self.start_time.elapsed().as_secs_f32();
        for p in &self.passes {
            if let Some(pipeline) = ctx.pipeline_store.get(p.pipeline_handle) {
                ctx.command_buffer.bind_pipeline(&pipeline);
                let push_constants = PushConstants {
                    time: current_time,
                    in_image: p.in_images.first().map(|&x| x as i32).unwrap_or(-1),
                    out_image: p.out_images.first().map(|&x| x as i32).unwrap_or(-1),
                };
                ctx.command_buffer.push_constants(&pipeline, vk::ShaderStageFlags::COMPUTE, 0, &bytemuck::cast_slice(std::slice::from_ref(&push_constants)));
                ctx.command_buffer.bind_push_descriptor_images(
                    &pipeline,
                    &image_resources.iter().map(|r| {
                        &r.image
                    }).collect::<Vec<&Image>>()
                );

                match p.dispatches {
                    DispatchConfig::FullScreen => {
                        let width = image_resources.first().unwrap().image.width();
                        let height = image_resources.first().unwrap().image.width();
                        let dispatches = UVec3::new(
                            (width as f32 / self.workgroup_size as f32).ceil() as u32,
                            (height as f32 / self.workgroup_size as f32).ceil() as u32,
                            1
                        );
                        ctx.command_buffer.dispatch(dispatches.x, dispatches.y, dispatches.z);
                    },
                    DispatchConfig::Count(x, y, z) => {
                        ctx.command_buffer.dispatch(x, y, z);
                    }
                }
            }

            // TODO: Add synchronization between passes
        };

        self.sink.as_ref().map(|sink| {
            let seekhead = sink.get_pos();
            let render_time = self.start_time.elapsed();

            if seekhead.abs_diff(render_time) > Duration::from_secs_f32(0.05) {
                _ = Sink::try_seek(sink, render_time);
            }
        });

        // Copy to target_image
        {
            let output_image = &image_resources.last().expect("No images found to output").image;

            ctx.command_buffer.image_barrier(
                &output_image,
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::TRANSFER_READ
            );

            // Transition the target image
            ctx.command_buffer.image_barrier(
                target_image,
                src_layout,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::NONE,
                vk::AccessFlags::TRANSFER_WRITE
            );

            ctx.command_buffer.clear_color_image(
                target_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                [0.0, 0.0, 0.0, 1.0]
            );

            // Use a blit, as a copy doesn't synchronize properly to the targetimage on MoltenVK

            ctx.command_buffer.blit_image(
                output_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit::default()
                    .src_offsets([
                        Offset3D::default(),
                        Offset3D::default().x(output_image.width() as i32).y(output_image.height() as i32).z(1)
                    ])
                    .dst_offsets([
                        Offset3D::default(),
                        Offset3D::default().x(output_image.width() as i32).y(output_image.height() as i32).z(1)
                    ])
                    .src_subresource(
                        ImageSubresourceLayers::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(0)
                    )
                    .dst_subresource(
                        ImageSubresourceLayers::default()
                            .aspect_mask(ImageAspectFlags::COLOR)
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(0)
                    )
                ],
                vk::Filter::NEAREST,
            );

            // Transfer back to default states
            ctx.command_buffer.image_barrier(
                &target_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                dst_layout,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::NONE
            );

            ctx.command_buffer.image_barrier(
                output_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::NONE
            );
        }
    }

    fn create_image_resources(device: &Device, allocator: &mut Allocator, draw_config: &DrawConfig, width: u32, height: u32) -> Vec<ImageResource> {
        let image_resources = draw_config.images.iter().map(|c| {
            let image = Image::new(
                device,
                allocator,
                width,
                height,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST
            );

            ImageResource {
                image,
                clear: c.clear.clone(),
            }
        }).collect::<Vec<ImageResource>>();

        image_resources
    }
}

impl GuiComponent for DrawOrchestrator {
    fn gui(&mut self, _: &GuiSystem, context: &Context) {
        TopBottomPanel::top("top").show(context, |ui| {
            menu::bar(ui, |ui| {
                ui.menu_button("Export..", |ui| {
                    // ui.label("Width");
                    // ui.add(egui::TextEdit::singleline(&mut self.image_export.width_text));
                    // ui.label("Height");
                    // ui.add(egui::TextEdit::singleline(&mut self.image_export.height_text));
                    ui.label("Filename");
                    ui.add(cen::egui::TextEdit::singleline(&mut self.image_export.filename));
                    ui.horizontal(|ui| {
                        ui.add(cen::egui::DragValue::new(&mut self.image_export.width));
                        ui.add(cen::egui::DragValue::new(&mut self.image_export.height));
                    });
                    if ui.button("Save").clicked() {
                        self.image_export.do_export = true;
                    }
                });
            });
        });
    }
}

impl RenderComponent for DrawOrchestrator {
    fn render(&mut self, ctx: &mut RenderContext) {

        if self.image_export.do_export {
            self.export(ctx, self.image_export.width, self.image_export.height);
            self.image_export.do_export = false;
        }

        // Recreate resources if needed
        let width = self.image_resources.first().unwrap().image.width();
        let height = self.image_resources.first().unwrap().image.height();
        if width != ctx.swapchain_image.width() || height != ctx.swapchain_image.height() {
            self.image_resources = Self::create_image_resources(ctx.device, ctx.allocator, &self.draw_config, ctx.swapchain_image.width(), ctx.swapchain_image.height());
        }

        self.do_render(ctx, &self.image_resources, ctx.swapchain_image, ImageLayout::PRESENT_SRC_KHR, ImageLayout::PRESENT_SRC_KHR);
    }
}
