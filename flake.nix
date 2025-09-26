{
  description = "Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust toolchain
            rust-bin.stable.latest.default
            
            # Development tools
            rust-analyzer
            cargo-watch
            cargo-edit

			# GPU
			vulkan-loader
			vulkan-headers
			vulkan-tools
			vulkan-validation-layers
			glslang
			spirv-tools

			# Wayland
			wayland
			wayland-protocols
            libxkbcommon
            
			# For building shaderc
			cmake
			python3

			# Audio
            alsa-lib
            alsa-lib.dev
            alsa-utils
            alsa-tools
          ];

	  # Set library paths
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.wayland
            pkgs.libxkbcommon
            pkgs.vulkan-loader
          ];

		  shellHook = ''
				export VK_LAYER_PATH="${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d"
				export LD_LIBRARY_PATH="${pkgs.wayland}/lib:${pkgs.libxkbcommon}/lib:${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH"
          '';

  		  PKG_CONFIG_PATH = "${pkgs.alsa-lib.dev}/lib/pkgconfig:${pkgs.jack2}/lib/pkgconfig";
          ALSA_PCM_CARD = "default";
          ALSA_PCM_DEVICE = "0";

          # Environment variables
          RUST_SRC_PATH = pkgs.rustPlatform.rustLibSrc;
        };
      });
}
