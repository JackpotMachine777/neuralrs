use std::sync::Arc;
use std::sync::OnceLock;

use cudarc::driver::{CudaContext, CudaModule, CudaStream};
use cudarc::nvrtc::compile_ptx;

#[cfg(feature = "cublas")]
use cudarc::cublas::CudaBlas;

struct Backend {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    #[cfg(feature = "cublas")]
    blas: CudaBlas,
}

static BACKEND: OnceLock<Backend> = OnceLock::new();

fn get() -> &'static Backend {
    BACKEND.get_or_init(|| {
        let ctx = CudaContext::new(0).expect("no usable CUDA device at index 0");
        let stream = ctx.default_stream();
        #[cfg(feature = "cublas")]
        let blas = CudaBlas::new(stream.clone()).expect("failed to create cuBLAS handle");

        Backend { ctx, stream, #[cfg(feature = "cublas")] blas, }
    })
}

pub fn stream() -> &'static Arc<CudaStream> {
    &get().stream
}

#[cfg(feature = "cublas")]
pub fn blas() -> &'static CudaBlas {
    &get().blas
}

pub fn compile(src: &str) -> Arc<CudaModule> {
    let ptx = compile_ptx(src).expect("CUDA kernel compilation (NVRTC) failed");
    get().ctx.load_module(ptx).expect("loading CUDA module failed")
}