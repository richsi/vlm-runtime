// compile .cu into .a

fn main() {
  println!("cargo:rerun-if-changed=kernels/*");

  // invoke nvcc
  cc::Build::new()
    .cuda(true)
    .flag("-cudart=shared")
    .flag("-ccbin=g++-12")
    .file("kernels/add.cu")
    .file("kernels/silu.cu")
    .compile("libkernels.a");

  println!("cargo:rustc-link-lib=cudart");
  println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
  println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
}