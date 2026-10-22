//! The exported weights (`sheetsage2_refined.export_cpu`): one safetensors file, memory-mapped for the life of the
//! process, so the weights are read from the page cache instead of copied onto the heap.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::Result;

pub const PANEL: usize = 16;

#[derive(Clone, Copy)]
pub enum Panels {
    F32(&'static [f32]),
    F16(&'static [u16]),
    I8(&'static [i8]),
}

/// A linear layer `y = W x + b`, `W` `[n, k]` stored as panels `[ceil(n / 16), k, 16]`; int8 weights carry a scale
/// per output channel.
pub struct Linear {
    pub n: usize,
    pub k: usize,
    pub w: Panels,
    pub scale: Option<&'static [f32]>,
    pub bias: &'static [f32],
}

impl Linear {
    pub fn panels(&self) -> usize {
        self.n.div_ceil(PANEL)
    }
}

pub struct Weights {
    map: &'static Mmap,
    tensors: HashMap<String, (Dtype, Vec<usize>, std::ops::Range<usize>)>,
    pub metadata: HashMap<String, String>,
}

impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let map: &'static Mmap = Box::leak(Box::new(unsafe { Mmap::map(&file)? }));
        let (header_len, header) = SafeTensors::read_metadata(&map[..])?;
        let start = 8 + header_len;
        let mut tensors = HashMap::new();
        for (name, info) in header.tensors() {
            let (a, b) = info.data_offsets;
            tensors.insert(name.clone(), (info.dtype, info.shape.clone(), start + a..start + b));
        }
        let metadata = header.metadata().clone().unwrap_or_default();
        if metadata.get("format").map(String::as_str) != Some("sheetsage2-cpu/1") {
            return Err(format!(
                "{}: not a sheetsage2-cpu/1 file (export it with sheetsage2_refined.export_cpu)",
                path.display()
            )
            .into());
        }
        Ok(Self { map, tensors, metadata })
    }

    /// Drop this process's resident pages of the file (Linux `MADV_DONTNEED`): after the encoder's weights went to
    /// the GPU, only the heads' small tensors are read again (and come back from the page cache when they are).
    pub fn release_pages(&self) {
        #[cfg(target_os = "linux")]
        unsafe {
            libc::madvise(self.map.as_ptr() as *mut libc::c_void, self.map.len(), libc::MADV_DONTNEED);
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn raw(&self, name: &str, dtype: Dtype) -> Result<(&'static [u8], &[usize])> {
        let (d, shape, range) = self.tensors.get(name).ok_or_else(|| format!("missing tensor {name}"))?;
        if *d != dtype {
            return Err(format!("{name}: expected {dtype:?}, found {d:?}").into());
        }
        Ok((&self.map[range.clone()], shape))
    }

    /// The bytes as `T`s (`f32`, `u16` or `i8`), copied only if the file left them misaligned (export_cpu does not).
    fn cast<T: Copy>(bytes: &'static [u8], name: &str) -> Result<&'static [T]> {
        let size = std::mem::size_of::<T>();
        if !bytes.len().is_multiple_of(size) {
            return Err(format!("{name}: {} bytes is not a whole number of elements", bytes.len()).into());
        }
        let count = bytes.len() / size;
        if (bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<T>()) {
            return Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, count) });
        }
        let mut owned: Vec<T> = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), owned.as_mut_ptr() as *mut u8, bytes.len());
            owned.set_len(count);
        }
        Ok(Box::leak(owned.into_boxed_slice()))
    }

    pub fn f32(&self, name: &str) -> Result<&'static [f32]> {
        let (bytes, _) = self.raw(name, Dtype::F32)?;
        Self::cast(bytes, name)
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.tensors.get(name).ok_or_else(|| format!("missing tensor {name}"))?.1.clone())
    }

    pub fn linear(&self, name: &str) -> Result<Linear> {
        let key = format!("{name}.w");
        let (dtype, shape, _) = self.tensors.get(&key).ok_or_else(|| format!("missing tensor {key}"))?;
        let (panels, k) = (shape[0], shape[1]);
        let w = match dtype {
            Dtype::F32 => Panels::F32(Self::cast(self.raw(&key, Dtype::F32)?.0, &key)?),
            Dtype::F16 => Panels::F16(Self::cast(self.raw(&key, Dtype::F16)?.0, &key)?),
            Dtype::I8 => Panels::I8(Self::cast(self.raw(&key, Dtype::I8)?.0, &key)?),
            other => return Err(format!("{key}: unsupported dtype {other:?}").into()),
        };
        let bias = self.f32(&format!("{name}.b"))?;
        let n = bias.len();
        if n.div_ceil(PANEL) != panels {
            return Err(format!("{name}: {n} outputs in {panels} panels").into());
        }
        let scale = if self.has(&format!("{name}.s")) { Some(self.f32(&format!("{name}.s"))?) } else { None };
        Ok(Linear { n, k, w, scale, bias })
    }
}
