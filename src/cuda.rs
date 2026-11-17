//! The CUDA driver API, opened at run time rather than linked.
//!
//! The driver library ships with the NVIDIA driver, not the toolkit, and
//! is the stable C interface underneath the CUDA runtime. Opening it with
//! dlopen or LoadLibrary keeps the executable free of any link-time
//! dependency on it: a machine without a driver still runs `--help` and
//! `--version`, and a search there fails with an error that says so,
//! where a linked dependency would stop the loader before main.
//!
//! Only the calls one kernel launch needs are bound. All of them are
//! resolved when the library is opened, so a driver too old to export one
//! is reported up front and not midway through a search. Several are
//! exported under a versioned name that cuda.h hides behind a macro; the
//! names here are the exported ones.
//!
//! This module is the function table and the error type, nothing more.
//! Context and module ownership, and every safe wrapper, live in gpu.rs.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::fmt;
use std::ptr;

use libloading::Library;

/// Zero is success; any other value is a CUDA_ERROR_* code.
pub type CUresult = c_int;
/// A device ordinal handle, as returned by cuDeviceGet.
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
/// A device address; 64 bits on every target this crate builds for.
pub type CUdeviceptr = u64;

/// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT from cuda.h.
pub const DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;

/// The name the driver installs its library under. The Linux name carries
/// the soname version because the unversioned libcuda.so is a development
/// symlink that a machine with only the driver does not have.
#[cfg(windows)]
const DRIVER_LIBRARY: &str = "nvcuda.dll";
#[cfg(not(windows))]
const DRIVER_LIBRARY: &str = "libcuda.so.1";

#[derive(Debug)]
pub enum CudaError {
    /// The driver library could not be opened.
    NoDriver(libloading::Error),
    /// The driver library does not export a function this build calls.
    MissingFunction {
        name: &'static str,
        source: libloading::Error,
    },
    /// A driver call returned a non-zero code. `description` is the
    /// driver's own text for it, from cuGetErrorString.
    Call {
        function: &'static str,
        code: CUresult,
        description: String,
    },
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::NoDriver(source) => write!(
                f,
                "cannot open {DRIVER_LIBRARY}: {source} (is the NVIDIA driver installed?)"
            ),
            CudaError::MissingFunction { name, source } => write!(
                f,
                "{DRIVER_LIBRARY} does not export {name}: {source} (the NVIDIA driver is too old)"
            ),
            CudaError::Call {
                function,
                code,
                description,
            } => write!(f, "{function} failed with CUDA error {code}: {description}"),
        }
    }
}

impl std::error::Error for CudaError {}

/// The driver's functions, as plain function pointers.
///
/// `extern "system"` is the driver's CUDAAPI: stdcall on 32-bit Windows
/// and the ordinary C convention everywhere else.
pub struct Driver {
    pub cu_init: unsafe extern "system" fn(flags: c_uint) -> CUresult,
    pub cu_get_error_string:
        unsafe extern "system" fn(error: CUresult, text: *mut *const c_char) -> CUresult,
    pub cu_device_get: unsafe extern "system" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    pub cu_device_get_attribute: unsafe extern "system" fn(
        value: *mut c_int,
        attribute: c_int,
        device: CUdevice,
    ) -> CUresult,
    pub cu_device_primary_ctx_retain:
        unsafe extern "system" fn(context: *mut CUcontext, device: CUdevice) -> CUresult,
    pub cu_device_primary_ctx_release: unsafe extern "system" fn(device: CUdevice) -> CUresult,
    pub cu_ctx_set_current: unsafe extern "system" fn(context: CUcontext) -> CUresult,
    pub cu_ctx_synchronize: unsafe extern "system" fn() -> CUresult,
    pub cu_module_load_data:
        unsafe extern "system" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    pub cu_module_unload: unsafe extern "system" fn(module: CUmodule) -> CUresult,
    pub cu_module_get_function: unsafe extern "system" fn(
        function: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult,
    pub cu_module_get_global: unsafe extern "system" fn(
        address: *mut CUdeviceptr,
        bytes: *mut usize,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult,
    pub cu_memcpy_htod:
        unsafe extern "system" fn(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult,
    pub cu_memcpy_dtoh:
        unsafe extern "system" fn(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult,
    pub cu_launch_kernel: unsafe extern "system" fn(
        function: CUfunction,
        grid_x: c_uint,
        grid_y: c_uint,
        grid_z: c_uint,
        block_x: c_uint,
        block_y: c_uint,
        block_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
    /// Keeps the library mapped for as long as the pointers above exist.
    _library: Library,
}

/// Resolve one exported function to a pointer of type `T`.
///
/// # Safety
/// `T` must be the function pointer type matching the driver's prototype
/// for `name`; a symbol carries no type, so nothing checks that it is.
unsafe fn resolve<T: Copy>(library: &Library, name: &'static str) -> Result<T, CudaError> {
    match library.get::<T>(name.as_bytes()) {
        Ok(symbol) => Ok(*symbol),
        Err(source) => Err(CudaError::MissingFunction { name, source }),
    }
}

impl Driver {
    /// Open the driver library and resolve every function in the table.
    pub fn load() -> Result<Self, CudaError> {
        // SAFETY: opening a library runs its initialisers; the NVIDIA
        // driver library is trusted to be what its name says.
        let library = unsafe { Library::new(DRIVER_LIBRARY) }.map_err(CudaError::NoDriver)?;
        // SAFETY: each field's type is the prototype cuda.h declares for
        // the exported name beside it.
        unsafe {
            Ok(Driver {
                cu_init: resolve(&library, "cuInit")?,
                cu_get_error_string: resolve(&library, "cuGetErrorString")?,
                cu_device_get: resolve(&library, "cuDeviceGet")?,
                cu_device_get_attribute: resolve(&library, "cuDeviceGetAttribute")?,
                cu_device_primary_ctx_retain: resolve(&library, "cuDevicePrimaryCtxRetain")?,
                cu_device_primary_ctx_release: resolve(&library, "cuDevicePrimaryCtxRelease_v2")?,
                cu_ctx_set_current: resolve(&library, "cuCtxSetCurrent")?,
                cu_ctx_synchronize: resolve(&library, "cuCtxSynchronize")?,
                cu_module_load_data: resolve(&library, "cuModuleLoadData")?,
                cu_module_unload: resolve(&library, "cuModuleUnload")?,
                cu_module_get_function: resolve(&library, "cuModuleGetFunction")?,
                cu_module_get_global: resolve(&library, "cuModuleGetGlobal_v2")?,
                cu_memcpy_htod: resolve(&library, "cuMemcpyHtoD_v2")?,
                cu_memcpy_dtoh: resolve(&library, "cuMemcpyDtoH_v2")?,
                cu_launch_kernel: resolve(&library, "cuLaunchKernel")?,
                _library: library,
            })
        }
    }

    /// Turn a driver return code into a Result, naming the call that
    /// produced it and attaching the driver's description of the code.
    pub fn check(&self, function: &'static str, code: CUresult) -> Result<(), CudaError> {
        if code == 0 {
            return Ok(());
        }
        let mut text: *const c_char = ptr::null();
        // SAFETY: cuGetErrorString writes a pointer to a static string, or
        // leaves `text` null for a code it does not know. It needs no
        // initialised driver or current context.
        let description = unsafe {
            if (self.cu_get_error_string)(code, &mut text) == 0 && !text.is_null() {
                CStr::from_ptr(text).to_string_lossy().into_owned()
            } else {
                String::from("unrecognised error code")
            }
        };
        Err(CudaError::Call {
            function,
            code,
            description,
        })
    }
}
