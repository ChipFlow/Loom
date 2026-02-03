//! this build script compiles GEM kernels
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    println!("cargo:rerun-if-changed=csrc");

    #[cfg(feature = "cuda")] {
        println!("Building CUDA source files for GEM...");
        let csrc_headers = ucc::import_csrc();
        let mut cl_cuda = ucc::cl_cuda();
        cl_cuda.ccbin(false);
        cl_cuda.flag("-lineinfo");
        cl_cuda.flag("-maxrregcount=128");
        cl_cuda.debug(false).opt_level(3)
            .include(&csrc_headers)
            .files(["csrc/kernel_v1.cu"]);
        cl_cuda.compile("gemcu");
        println!("cargo:rustc-link-lib=static=gemcu");
        println!("cargo:rustc-link-lib=dylib=cudart");
        ucc::bindgen(["csrc/kernel_v1.cu"], "kernel_v1.rs");
        ucc::export_csrc();
        ucc::make_compile_commands(&[&cl_cuda]);
    }

    #[cfg(feature = "metal")] {
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
