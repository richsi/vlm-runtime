use std::ffi::c_void;
use std::ptr;

// define cuda functions
unsafe extern "C" {
  fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> i32;
  fn cudaFree(devPtr: *mut c_void) -> i32;
  fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
}

/*
*mud c_void = void*
*const f32 = const float*
ptr::null_mtr() = nullptr
*/

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

// initialize tensor struct
pub struct Tensor {
  ptr: *mut c_void,
  pub shape: Vec<usize>,
  size: usize, // number of bytes
}

// implement tensor
impl Tensor {
  // constructor
  pub fn new(shape: Vec<usize>) -> Self {
    let num_elements: usize = shape.iter().product();
    let size = num_elements * 4; // assuming f32 (4 bytes)
    let mut ptr: *mut c_void = ptr::null_mut();

    unsafe {
      // ask GPU for memory
      let status = cudaMalloc(&mut ptr, size);
      assert_eq!(status, 0, "CUDA Malloc failed!");
    }

    Tensor { ptr, shape, size }
  }

  // Loads from host to device (CPU -> GPU)
  pub fn load(&mut self, data: &[f32]) {
    assert_eq!(data.len() * 4, self.size);
    unsafe {
      cudaMemcpy(
        self.ptr,
        data.as_ptr() as *const c_void,
        self.size,
        CUDA_MEMCPY_HOST_TO_DEVICE
      );
    }
  }

  // Reads from device to host (GPU -> CPU)
  pub fn read(&self) -> Vec<f32> {
    let num_elements = self.size / 4;
    let mut host_data = vec![0.0f32; num_elements];
    
    unsafe {
      cudaMemcpy(
        host_data.as_mut_ptr() as *mut c_void, // dest
        self.ptr, // source
        self.size,
        CUDA_MEMCPY_DEVICE_TO_HOST
      );
    }

    host_data // return
  }

  pub fn ptr(&self) -> *const f32 {
    self.ptr as *const f32
  }

  pub fn mut_ptr(&self) -> *mut f32 {
    self.ptr as *mut f32
  }


  // helper to get raw pointers
}

// automatic deallocation 
impl Drop for Tensor {
  fn drop(&mut self) {
    unsafe {
      cudaFree(self.ptr);
    }
  }
}
