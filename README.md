# Kiyo
![build](https://github.com/n-e-l/kiyo/actions/workflows/rust.yml/badge.svg)
[![crate](https://img.shields.io/crates/v/kiyo)](https://crates.io/crates/kiyo/)  

## What is Kiyo?
A lightweight compute shader engine using [ash](https://github.com/ash-rs/ash).

Kiyo provides a simple configuration interface to run compute shaders. The following features are implemented:
- Multiple subsequent compute passes
- Shared storage images between them
- GLSL compile logging
- Shader hot-reloading
- PNG exporting at custom resolutions

For any feedback or requests you are very welcome to create issues or contact me directly!

You can find examples in [./examples/](examples/) and in my [toy project repository](https://github.com/angelocarly/kiyo-projects).

## Shader environment variables
These variables are accessible in the shader and provided by Kiyo itself, do not overwrite these as bugs will be introduced.
- `NUM_IMAGES` - The amount of accessible storage images.
- `WORKGROUP_SIZE` - The workgroup size at which the shaders should run.

## Building & running

Make sure you have the [Vulkan SDK](https://vulkan.lunarg.com) installed.  
Then build and run `kiyo`:
```
git clone https://github.com/angelocarly/kiyo.git
cd kiyo
cargo run --example simple-render
```

## Quickstart

You can quickly setup a small Rust project using kiyo with [this](https://github.com/angelocarly/kiyo-template) template repository.

## GPU debugging

### Windows & Linux
Renderdoc!

### Mac
Mac only has XCode's Metal debugger. In order to use it you need to provide the following environment variables:
```bash
VULKAN_SDK=$HOME/VulkanSDK/<version>/macOS
DYLD_FALLBACK_LIBRARY_PATH=$VULKAN_SDK/lib
VK_ICD_FILENAMES=$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json
VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
```

Then you should be able to launch your kiyo application and capture a frame.  
[This video](https://www.youtube.com/watch?v=uNB4RMZg1AM) does a nice job explaining the process.

## References
- [sound-shader](https://github.com/ytanimura/sound-shader) - Cpal wrapper code and shader audio inspiration

## Libraries
- [cen](https://github.com/angelocarly/cen) - Window management and Vulkan wrappers
- [ash](https://github.com/ash-rs/ash) - Vulkan bindings
