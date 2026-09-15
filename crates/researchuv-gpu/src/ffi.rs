//! CUDA driver API, loaded at runtime — no link-time toolkit dependency.
//!
//! `nvcuda.dll` (Windows) / `libcuda.so.1` (Unix) is resolved with
//! `LoadLibraryA`/`dlopen` on first use and the handful of driver entry
//! points we need are fetched by name (`_v2` 64-bit symbols preferred).
//! Machines without a CUDA driver — or without the PTX from the build step —
//! simply report "unavailable" and the engine falls back to its CPU solve.
//!
//! This module is the one place in the workspace that touches `unsafe`; the
//! safe wrapper ([`super::solver`]) owns every resource.

#![allow(non_snake_case)]

use std::ffi::c_void;

pub type CUresult = u32;
pub type CUdevice = i32;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUdeviceptr = u64;
pub type CUstream = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

type FnCuInit = unsafe extern "system" fn(flags: u32) -> CUresult;
type FnCuDeviceGet = unsafe extern "system" fn(*mut CUdevice, i32) -> CUresult;
type FnCuDevicePrimaryCtxRetain = unsafe extern "system" fn(*mut CUcontext, CUdevice) -> CUresult;
type FnCuCtxSetCurrent = unsafe extern "system" fn(CUcontext) -> CUresult;
type FnCuCtxSynchronize = unsafe extern "system" fn() -> CUresult;
type FnCuModuleLoadData = unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnCuModuleGetFunction =
    unsafe extern "system" fn(*mut CUfunction, CUmodule, *const u8) -> CUresult;
type FnCuMemAlloc = unsafe extern "system" fn(*mut CUdeviceptr, usize) -> CUresult;
pub type FnCuMemFree = unsafe extern "system" fn(CUdeviceptr) -> CUresult;
type FnCuMemcpyHtoD = unsafe extern "system" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type FnCuMemcpyDtoH = unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type FnCuLaunchKernel = unsafe extern "system" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;
type FnCuGetErrorString = unsafe extern "system" fn(CUresult, *mut *const i8) -> CUresult;

/// The resolved driver entry points.
#[allow(clippy::type_complexity)]
pub struct Cuda {
    pub cuInit: FnCuInit,
    pub cuDeviceGet: FnCuDeviceGet,
    pub cuDevicePrimaryCtxRetain: FnCuDevicePrimaryCtxRetain,
    pub cuCtxSetCurrent: FnCuCtxSetCurrent,
    pub cuCtxSynchronize: FnCuCtxSynchronize,
    pub cuModuleLoadData: FnCuModuleLoadData,
    pub cuModuleGetFunction: FnCuModuleGetFunction,
    pub cuMemAlloc: FnCuMemAlloc,
    pub cuMemFree: FnCuMemFree,
    pub cuMemcpyHtoD: FnCuMemcpyHtoD,
    pub cuMemcpyDtoH: FnCuMemcpyDtoH,
    pub cuLaunchKernel: FnCuLaunchKernel,
    #[allow(dead_code)]
    pub cuGetErrorString: FnCuGetErrorString,
}

/// Load the driver library and resolve every entry point (all or nothing).
pub fn load() -> Option<Cuda> {
    struct Resolver {
        handle: *mut c_void,
        ok: bool,
    }
    impl Resolver {
        /// First hit wins (callers list the `_v2` name before the plain one).
        fn sym(&mut self, names: &[&[u8]]) -> *mut c_void {
            if !self.ok {
                return std::ptr::null_mut();
            }
            for n in names {
                // SAFETY: the handle is a live library and the name is
                // NUL-terminated; both come from this module's callers.
                let p = unsafe { library_symbol(self.handle, n.as_ptr()) };
                if !p.is_null() {
                    return p;
                }
            }
            self.ok = false;
            std::ptr::null_mut()
        }
    }

    // SAFETY: this whole function resolves the driver's documented,
    // ABI-stable entry points and transmutes them to matching fn types.
    unsafe {
        let handle = open_library()?;
        let mut res = Resolver { handle, ok: true };
        macro_rules! resolve {
            ($ty:ty, $($name:literal),+ $(,)?) => {{
                let p = res.sym(&[$($name),+]);
                if p.is_null() {
                    return None;
                }
                std::mem::transmute::<*mut c_void, $ty>(p)
            }};
        }
        Some(Cuda {
            cuInit: resolve!(FnCuInit, b"cuInit\0"),
            cuDeviceGet: resolve!(FnCuDeviceGet, b"cuDeviceGet\0"),
            cuDevicePrimaryCtxRetain: resolve!(FnCuDevicePrimaryCtxRetain, b"cuDevicePrimaryCtxRetain\0"),
            cuCtxSetCurrent: resolve!(FnCuCtxSetCurrent, b"cuCtxSetCurrent\0"),
            cuCtxSynchronize: resolve!(FnCuCtxSynchronize, b"cuCtxSynchronize\0"),
            cuModuleLoadData: resolve!(FnCuModuleLoadData, b"cuModuleLoadData_v2\0", b"cuModuleLoadData\0"),
            cuModuleGetFunction: resolve!(FnCuModuleGetFunction, b"cuModuleGetFunction\0"),
            cuMemAlloc: resolve!(FnCuMemAlloc, b"cuMemAlloc_v2\0"),
            cuMemFree: resolve!(FnCuMemFree, b"cuMemFree_v2\0"),
            cuMemcpyHtoD: resolve!(FnCuMemcpyHtoD, b"cuMemcpyHtoD_v2\0"),
            cuMemcpyDtoH: resolve!(FnCuMemcpyDtoH, b"cuMemcpyDtoH_v2\0"),
            cuLaunchKernel: resolve!(FnCuLaunchKernel, b"cuLaunchKernel\0"),
            cuGetErrorString: resolve!(FnCuGetErrorString, b"cuGetErrorString\0"),
        })
    }
}

#[cfg(windows)]
unsafe fn open_library() -> Option<*mut c_void> {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const u8) -> *mut c_void;
    }
    let h = LoadLibraryA(b"nvcuda.dll\0".as_ptr());
    (!h.is_null()).then_some(h)
}

#[cfg(windows)]
unsafe fn library_symbol(handle: *mut c_void, name: *const u8) -> *mut c_void {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetProcAddress(handle: *mut c_void, name: *const u8) -> *mut c_void;
    }
    GetProcAddress(handle, name)
}

#[cfg(not(windows))]
unsafe fn open_library() -> Option<*mut c_void> {
    const RTLD_NOW: i32 = 2;
    extern "C" {
        fn dlopen(filename: *const i8, flag: i32) -> *mut c_void;
    }
    let h = dlopen(b"libcuda.so.1\0".as_ptr() as *const i8, RTLD_NOW);
    (!h.is_null()).then_some(h)
}

#[cfg(not(windows))]
unsafe fn library_symbol(handle: *mut c_void, name: *const u8) -> *mut c_void {
    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const i8) -> *mut c_void;
    }
    dlsym(handle, name as *const i8)
}
