use std::collections::HashMap;
use std::mem::size_of;
use cen::ash::vk;
use cen::ash::vk::{BufferImageCopy, BufferUsageFlags, DeviceSize, Extent3D, ImageAspectFlags, ImageLayout, ImageSubresourceLayers, Offset3D, WriteDescriptorSet};
use bytemuck::{Pod, Zeroable};
use cen::graphics::pipeline_store::{PipelineConfig, PipelineKey};
use cen::graphics::renderer::{RenderComponent, RenderContext};
use cen::vulkan::{Allocator, Buffer, DescriptorSetLayout, Device, Image, PipelineErr};
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
use cen::app::gui::{GuiComponent, GuiHandler};
use cen::egui;
use cen::egui::{emath, Context, ImageSource, Pos2, Rect, Scene, TextureId, TopBottomPanel, Widget};
use cen::egui::load::SizedTexture;
use cen::gpu_allocator::MemoryLocation;
use egui_dock::{DockArea, DockState, NodeIndex, Style};
use egui_dock::tab_viewer::OnCloseResponse;
use crate::app::png::{write_png_image};

type Tab = String;

#[derive(Copy)]
#[derive(Clone)]
pub enum DispatchConfig
{
    Count( u32, u32, u32 ),
    /**
     * Dispatch *at least* as many shader invocations as there are pixels (x,y) in the image.
     */
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

#[derive(Clone, Copy)]
pub enum ClearConfig {
    None,
    Color(f32,f32,f32),
}

#[derive(Clone, Copy)]
pub enum AtomicClearConfig {
    None,
    Color(u32,u32,u32),
}

#[derive(Clone)]
pub struct ImageConfig {
    pub clear: ClearConfig,
}

pub struct DrawConfig {
    pub width: u32,
    pub height: u32,
    pub passes: Vec<Pass>,
    pub images: Vec<ImageConfig>,
    pub atomic_image: AtomicClearConfig,
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

pub struct AtomicImageResource {
    pub images: Vec<Image>,
    pub clear: AtomicClearConfig,
}

struct ImgExport {
    filename: String,
    do_export: bool,
}

/**
 *  Contains all render related structures relating to a config.
 */
pub struct DrawOrchestrator {
    draw_config: DrawConfig,
    #[allow(dead_code)]
    audio_config: AudioConfig,
    #[allow(dead_code)]
    audio_stream: Option<OutputStream>,
    sink: Option<Sink>,
    pub compute_descriptor_set_layout: DescriptorSetLayout,
    pub image_resources: Vec<ImageResource>,
    pub counter_images: AtomicImageResource,
    pub passes: Vec<ShaderPass>,
    image_export: ImgExport,
    workgroup_size: u32,
    start_time: std::time::Instant,
    texture_id: Option<TextureId>,
    scene_rect: emath::Rect,
    dock_state: DockState<String>,
    render_controls: RenderControls,
    first_frame: bool,
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
            // General purpose images
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(image_count)
                .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT),
            // Atomic add images
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(3)
                .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::FRAGMENT)
        ];
        let compute_descriptor_set_layout = DescriptorSetLayout::new_push_descriptor(
            &ctx.device,
            layout_bindings
        );

        // Images
        let image_resources = Self::create_image_resources(&ctx.device, &mut ctx.allocator, &draw_config, draw_config.width, draw_config.height);
        let counter_images = AtomicImageResource {
            images:  (0..3).map(|_| {
                Image::builder(ctx.device, &mut ctx.allocator)
                    .width(draw_config.width)
                    .height(draw_config.height)
                    .format(vk::Format::R32_UINT)
                    .image_usage_flags(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)
                    .build()
            }).collect::<Vec<Image>>(),
            clear: draw_config.atomic_image
        };

        // let texture_id = Some(ctx.gui_system.create_texture(&image_resources.first().unwrap().image));

        // Shader
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

        let scene_rect = emath::Rect { min: Pos2::new(0., 0.), max: Pos2::new(draw_config.width as f32, draw_config.height as f32) };

        let mut dock_state = DockState::new(vec!["content".to_owned()]);

        let [a, b] =
            dock_state.main_surface_mut()
                .split_left(NodeIndex::root(), 0.1, vec!["tools".to_string()]);

        let render_controls = RenderControls {
            running: false,
            step: false
        };

        Self {
            workgroup_size: 32,
            draw_config,
            audio_config,
            audio_stream,
            sink,
            compute_descriptor_set_layout,
            image_resources,
            counter_images,
            passes,
            image_export: ImgExport {
                do_export: false,
                filename: "output".to_string(),
            },
            start_time: std::time::Instant::now(),
            texture_id: None,
            scene_rect,
            dock_state,
            render_controls,
            first_frame: true
        }
    }

    fn export(&mut self, ctx: &mut RenderContext, width: u32, height: u32) {

        info!("Exporting...");
        let buffer = Buffer::new(
            &ctx.device,
            &mut ctx.allocator,
            MemoryLocation::GpuToCpu,
            (size_of::<u8>() as u32 * 4 * width * height) as DeviceSize,
            BufferUsageFlags::STORAGE_BUFFER | BufferUsageFlags::TRANSFER_DST
        );

        let output_image = &self.image_resources.last().unwrap().image;
        ctx.command_buffer.image_barrier(
            output_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::NONE,
            vk::AccessFlags::TRANSFER_WRITE
        );

        ctx.command_buffer.copy_image_to_buffer(
            output_image,
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

        ctx.command_buffer.image_barrier(
            output_image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::NONE
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
    fn do_render(&self, ctx: &mut RenderContext, image_resources: &Vec<ImageResource>) {

        // Clear all images with a clear config
        {
            for i in &self.counter_images.images {
                ctx.command_buffer.image_barrier(
                    &i,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::NONE,
                    vk::AccessFlags::TRANSFER_WRITE
                );

                match &self.counter_images.clear {
                    AtomicClearConfig::None => {},
                    AtomicClearConfig::Color(r, g, b) => {
                        ctx.command_buffer.clear_color_image_u32(
                            &i,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            [*r, *g, *b, 0]
                        );
                    }
                }

                ctx.command_buffer.image_barrier(
                    &i,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_WRITE
                );
            }
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

                let image_bindings = self.image_resources.iter().map(|image| {
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(image.image.image_view())
                        .sampler(image.image.sampler())
                }).collect::<Vec<vk::DescriptorImageInfo>>();

                let counter_image_bindings = self.counter_images.images.iter().map(|image| {
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::GENERAL)
                        .image_view(image.image_view())
                        .sampler(image.sampler())
                }).collect::<Vec<vk::DescriptorImageInfo>>();

                let write_descriptor_sets = [
                    WriteDescriptorSet::default()
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&image_bindings),
                    WriteDescriptorSet::default()
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&counter_image_bindings),
                ];

                ctx.command_buffer.push_descriptor_set(
                    &pipeline,
                    0,
                    write_descriptor_sets.as_slice()
                );

                for image in &self.image_resources { ctx.command_buffer.track(&image.image); }
                for image in &self.counter_images.images { ctx.command_buffer.track(image); }

                match p.dispatches {
                    DispatchConfig::FullScreen => {
                        let width = image_resources.first().unwrap().image.width();
                        let height = image_resources.first().unwrap().image.height();
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

        // Transfer to be readable by egui
        for i in image_resources {
            ctx.command_buffer.image_barrier(
                &i.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::NONE
            );
        }
    }

    fn create_image_resources(device: &Device, allocator: &mut Allocator, draw_config: &DrawConfig, width: u32, height: u32) -> Vec<ImageResource> {
        let image_resources = draw_config.images.iter().map(|c| {
            let image = Image::new_rgba(
                device,
                allocator,
                width,
                height,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED
            );

            ImageResource {
                image,
                clear: c.clear.clone(),
            }
        }).collect::<Vec<ImageResource>>();

        image_resources
    }
}

struct ViewPanel<'a> {
    image: &'a Image,
    scene_rect: &'a mut Rect,
    texture_id: &'a mut Option<TextureId>,
}

struct RenderControls {
    running: bool,
    step: bool
}

struct TabViewer<'a> {
    panel: ViewPanel<'a>,
    controls: &'a mut RenderControls,
}

impl<'a> egui_dock::TabViewer for TabViewer<'a> {
    type Tab = String;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        (&*tab).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        if tab == "content" {
            let im = &self.panel.image;
            let size = cen::egui::Vec2 { x: im.width() as f32, y: im.height() as f32 };
            if let Some(id) = self.panel.texture_id {
                Scene::new()
                    .zoom_range(0.2..=50.0)
                    .show(ui, &mut self.panel.scene_rect, |ui| {
                    egui::Image::new(ImageSource::Texture(SizedTexture {
                        id: *id,
                        size
                    })).ui(ui);
                });
            }
        } else if tab == "tools" {
            if ui.button("play").clicked() {
                self.controls.running = true;
            }
            if ui.button("pause").clicked() {
                self.controls.running = false;
            }
            self.controls.step =  ui.button("step").clicked();
        }
    }

    fn on_close(&mut self, _tab: &mut Self::Tab) -> OnCloseResponse {
        println!("Closed tab: {_tab}");
        OnCloseResponse::Close
    }
}

impl GuiComponent for DrawOrchestrator {
    fn gui(&mut self, gui: &mut GuiHandler, context: &Context) {

        TopBottomPanel::top("top").show(context, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("Export..", |ui| {
                    ui.label("Filename");
                    ui.add(cen::egui::TextEdit::singleline(&mut self.image_export.filename));
                    if ui.button("Save").clicked() {
                        self.image_export.do_export = true;
                    }
                });
                if ui.button("Reset view").clicked() {
                    self.scene_rect = emath::Rect { min: Pos2::new(0., 0.), max: Pos2::new(self.draw_config.width as f32, self.draw_config.height as f32) };
                };
            });
        });

        let image = &self.image_resources.last().unwrap().image;
        if self.texture_id.is_none() {
            self.texture_id = Some(gui.create_texture(image));
        }
        let panel_data = ViewPanel {
            image,
            scene_rect: &mut self.scene_rect,
            texture_id: &mut self.texture_id,
        };
        egui::CentralPanel::default().show(context, |ui| {
            DockArea::new(&mut self.dock_state)
                .style(Style::from_egui(ui.style().as_ref()))
                .show_inside(ui, &mut TabViewer {
                    panel: panel_data,
                    controls: &mut self.render_controls
                });
        });
    }
}

impl RenderComponent for DrawOrchestrator {
    fn render(&mut self, ctx: &mut RenderContext) {

        if self.first_frame || self.render_controls.step || self.render_controls.running {
            self.do_render(ctx, &self.image_resources);
            self.first_frame = false;
        }

        if self.image_export.do_export {
            self.export(ctx, self.draw_config.width, self.draw_config.height);
            self.image_export.do_export = false;
        }
    }
}
