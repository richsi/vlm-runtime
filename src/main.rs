mod tensor;
use tensor::Tensor;

unsafe extern "C" {
  fn launch_add(a: *const f32, b: *const f32, out: *mut f32, n: i32);
  fn launch_silu(input: *const f32, output: *mut f32, n: i32);
}

fn main() {
  let n = 5;

  // allocate and load
  let input_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
  let mut x = Tensor::new(vec![n]);

  x.load(&input_data);

  let mut out = Tensor::new(vec![n]);

  println!("Input: {:?}", input_data);

  unsafe {
    launch_silu(x.ptr(), out.mut_ptr(), n as i32);
  }

  let result = out.read();
  println!("Result: {:?}", &result);
}
