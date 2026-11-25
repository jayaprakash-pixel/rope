//! Read-only memory map of a whole file (Windows and Linux). The only `unsafe` in the model-file path:
//! the OS returns a pointer, and we expose it as a bounds-checked byte slice that lives as long as the map.
pub use imp::Map;

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;
    use std::io;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    type Handle = isize;
    const INVALID: Handle = -1;

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateFileW(name: *const u16, access: u32, share: u32, sa: *const c_void, disp: u32, flags: u32, tmpl: Handle) -> Handle;
        fn GetFileSizeEx(h: Handle, size: *mut i64) -> i32;
        fn CreateFileMappingW(h: Handle, sa: *const c_void, protect: u32, hi: u32, lo: u32, name: *const u16) -> Handle;
        fn MapViewOfFile(m: Handle, access: u32, hi: u32, lo: u32, bytes: usize) -> *mut c_void;
        fn UnmapViewOfFile(base: *const c_void) -> i32;
        fn CloseHandle(h: Handle) -> i32;
    }

    pub struct Map { base: *const u8, len: usize, file: Handle, map: Handle }

    impl Map {
        pub fn open(path: &Path) -> io::Result<Map> {
            let w: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
            // SAFETY: valid NUL-terminated wide path; every handle is checked and closed on every exit path.
            unsafe {
                let file = CreateFileW(w.as_ptr(), 0x8000_0000 /*GENERIC_READ*/, 1 /*FILE_SHARE_READ*/, std::ptr::null(),
                                       3 /*OPEN_EXISTING*/, 0x80 /*NORMAL*/, 0);
                if file == INVALID { return Err(io::Error::last_os_error()); }
                let mut size = 0i64;
                if GetFileSizeEx(file, &mut size) == 0 || size <= 0 {
                    let e = io::Error::last_os_error(); CloseHandle(file); return Err(e);
                }
                let map = CreateFileMappingW(file, std::ptr::null(), 2 /*PAGE_READONLY*/, 0, 0, std::ptr::null());
                if map == 0 { let e = io::Error::last_os_error(); CloseHandle(file); return Err(e); }
                let base = MapViewOfFile(map, 4 /*FILE_MAP_READ*/, 0, 0, 0) as *const u8;
                if base.is_null() { let e = io::Error::last_os_error(); CloseHandle(map); CloseHandle(file); return Err(e); }
                Ok(Map { base, len: size as usize, file, map })
            }
        }

        pub fn bytes(&self) -> &[u8] {
            // SAFETY: the view covers exactly `len` readable bytes and stays mapped until `self` is dropped.
            unsafe { std::slice::from_raw_parts(self.base, self.len) }
        }
    }

    impl Drop for Map {
        fn drop(&mut self) {
            // SAFETY: handles and view were created in `open` and are released exactly once.
            unsafe { UnmapViewOfFile(self.base as *const c_void); CloseHandle(self.map); CloseHandle(self.file); }
        }
    }
}

#[cfg(unix)]
mod imp {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::path::Path;

    extern "C" {
        fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
    }
    const PROT_READ: i32 = 1;
    const MAP_PRIVATE: i32 = 2;

    pub struct Map { base: *const u8, len: usize }

    impl Map {
        pub fn open(path: &Path) -> io::Result<Map> {
            let file = File::open(path)?;
            let len = file.metadata()?.len() as usize;
            if len == 0 { return Err(io::Error::new(io::ErrorKind::InvalidData, "empty file")); }
            // SAFETY: a valid open descriptor and its exact length; the result is checked (MAP_FAILED is -1).
            // The mapping stays valid after `file` is closed.
            let base = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
            if base as isize == -1 { return Err(io::Error::last_os_error()); }
            Ok(Map { base: base as *const u8, len })
        }

        pub fn bytes(&self) -> &[u8] {
            // SAFETY: the mapping covers exactly `len` readable bytes and stays mapped until `self` is dropped.
            unsafe { std::slice::from_raw_parts(self.base, self.len) }
        }
    }

    impl Drop for Map {
        fn drop(&mut self) {
            // SAFETY: created in `open`, released exactly once.
            unsafe { munmap(self.base as *mut c_void, self.len); }
        }
    }
}
