mod tensor;
use tensor::Tensor;

unsafe extern "C" {
  fn launch_add(a: *const f32, b: *const f32, out: *mut f32, n: i32);
}

fn main() {
  let n = 100;

  // allocate and load
  let mut a = Tensor::new(vec![n]);
  a.load(&vec![1.0; n]);

  let mut b = Tensor::new(vec![n]);
  b.load(&vec![2.0; n]);

  let mut out = Tensor::new(vec![n]);

  unsafe {
    println!("Launching kernel...");
    launch_add(a.ptr(), b.ptr(), out.mut_ptr(), n as i32);
  }

  let result = out.read();
  println!("Result: {:?}", &result[0..5]);
}
