//! this build script compiles GEM kernels
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("cargo:rerun-if-changed=csrc");

    // Build the C++ SPI flash model
    cc::Build::new()
        .cpp(true)
        .file("csrc/spiflash_model.cc")
        .include("csrc")
        .compile("spiflash_model");

    #[cfg(feature = "cuda")]
    {
        println!("Building CUDA source files for GEM...");
        let csrc_headers = ucc::import_csrc();
        let mut cl_cuda = ucc::cl_cuda();
        cl_cuda.ccbin(false);
        cl_cuda.flag("-lineinfo");
        cl_cuda.flag("-maxrregcount=128");
        cl_cuda
            .debug(false)
            .opt_level(3)
            .include(&csrc_headers)
            .files(["csrc/kernel_v1.cu"]);
        cl_cuda.compile("gemcu");
        println!("cargo:rustc-link-lib=static=gemcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
        ucc::bindgen(["csrc/kernel_v1.cu"], "kernel_v1.rs");
        ucc::export_csrc();
        ucc::make_compile_commands(&[&cl_cuda]);
    }

    #[cfg(feature = "hip")]
    {
        println!("Building HIP source files for GEM...");
        let csrc_headers = ucc::import_csrc();
        let mut cl_hip = ucc::cl_hip();
        cl_hip
            .debug(false)
            .opt_level(3)
            .include(&csrc_headers)
            .file("csrc/kernel_v1.hip.cpp");
        cl_hip.compile("gemhip");
        println!("cargo:rustc-link-lib=static=gemhip");
        // On AMD backend, link amdhip64; on NVIDIA backend, link cudart.
        // The kernel_v1.hip.cpp wrapper handles both via hipcc compilation.
        if std::env::var("HIP_PLATFORM").as_deref() == Ok("nvidia") {
            println!("cargo:rustc-link-lib=dylib=cudart");
            let cuda_path = std::env::var("CUDA_PATH")
                .unwrap_or_else(|_| "/usr/local/cuda".to_string());
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        } else {
            println!("cargo:rustc-link-lib=dylib=amdhip64");
            let rocm_path = std::env::var("ROCM_PATH")
                .unwrap_or_else(|_| "/opt/rocm".to_string());
            println!("cargo:rustc-link-search=native={}/lib", rocm_path);
        }
        println!("cargo:rerun-if-env-changed=HIP_PLATFORM");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        ucc::bindgen(["csrc/kernel_v1.hip.cpp"], "kernel_v1_hip.rs");
        ucc::export_csrc();
        ucc::make_compile_commands(&[&cl_hip]);
    }

    #[cfg(feature = "metal")]
    {
        println!("Building Metal shader for GEM...");
        // Compile Metal shader to metallib
        ucc::cl_metal()
            .file("csrc/kernel_v1.metal")
            .std_version("metal3.0")
            .macos_version_min("14.0")
            .compile("gem_metal");
        // METALLIB_PATH environment variable is set by the compile step
    }
}
