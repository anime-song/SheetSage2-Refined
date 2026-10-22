//! LoRA adapters applied at run time (`python -m sheetsage2_refined.export_cpu out.lora.safetensors --adapter ...`):
//! the base weights stay as they are and each adapted linear layer adds `(x A^T) B^T + db`, two thin products of
//! rank `R` (a few per cent of the layer's own work at the ranks LoRA uses).  An adapter is read into memory as
//! float16 (a few MiB to a few tens) and can be swapped or dropped between songs without touching the 1.2 GiB of
//! base weights.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use safetensors::{Dtype, SafeTensors};

use crate::Result;
use crate::weights::{Linear, PANEL, Panels};

pub const FORMAT: &str = "sheetsage2-cpu-lora/1";

/// One part of a layer's adapter: `up` `[n, r]` maps columns `low .. low + r` of `x A^T` to the layer's outputs
/// `row .. row + n` (q, k and v share a layer, and each has its own part).
pub struct Part {
    pub row: usize,
    pub low: usize,
    pub up: Linear,
}

/// One layer's adapter: `down` `[R, K]` (every part's rows stacked) and its parts, for a layer of `n` outputs.
pub struct Lora {
    pub n: usize,
    pub rank: usize,
    pub down: Linear,
    pub parts: Vec<Part>,
}

/// The adapters of one LoRA file, by engine layer name (`layers.3.attn.qkv`, `stages.1.layers.0.fc1`, ...).
pub struct Adapter {
    pub layers: HashMap<String, Arc<Lora>>,
    pub metadata: HashMap<String, String>,
}

/// Row-major `[n, k]` as panels `[ceil(n / 16), k, 16]`.
pub fn pack<T: Copy + Default>(w: &[T], n: usize, k: usize) -> Vec<T> {
    let mut out = vec![T::default(); n.div_ceil(PANEL) * k * PANEL];
    for j in 0..n {
        for p in 0..k {
            out[((j / PANEL) * k + p) * PANEL + j % PANEL] = w[j * k + p];
        }
    }
    out
}

fn linear(w: &[u16], n: usize, k: usize, bias: Vec<f32>) -> Linear {
    Linear { n, k, w: Panels::F16(pack(w, n, k).into()), scale: None, bias: bias.into(), lora: None }
}

impl Lora {
    /// From row-major float16 bits: `a` `[rank, k]`, and per part its first output, `b` `[n, r]` and `n`, in the
    /// order of `a`'s rows; `db` `[outputs]` (zero where no part writes).
    pub fn new(a: &[u16], rank: usize, parts: &[(usize, &[u16], usize)], db: &[f32]) -> Self {
        let k = a.len() / rank;
        assert_eq!(a.len(), rank * k);
        let mut low = 0;
        let parts = parts
            .iter()
            .map(|&(row, b, n)| {
                let r = b.len() / n;
                let bias = db[row..row + n].to_vec();
                let part = Part { row, low, up: linear(b, n, r, bias) };
                low += r;
                part
            })
            .collect::<Vec<_>>();
        assert_eq!(low, rank);
        Self { n: db.len(), rank, down: linear(a, rank, k, vec![0.0; rank]), parts }
    }

    pub fn bytes(&self) -> usize {
        let halves = self.rank * self.down.k + self.parts.iter().map(|p| p.up.n * p.up.k).sum::<usize>();
        2 * halves + 4 * self.n
    }
}

impl Adapter {
    pub fn open(path: &Path) -> Result<Self> {
        let data = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file = SafeTensors::deserialize(&data)?;
        let (_, header) = SafeTensors::read_metadata(&data)?;
        let metadata = header.metadata().clone().unwrap_or_default();
        if metadata.get("format").map(String::as_str) != Some(FORMAT) {
            return Err(format!(
                "{}: not a {FORMAT} file (write it with sheetsage2_refined.export_cpu --adapter)",
                path.display()
            )
            .into());
        }
        let tensor = |name: &str, dtype: Dtype| -> Result<(&[u8], Vec<usize>)> {
            let t = file.tensor(name).map_err(|_| format!("{}: missing tensor {name}", path.display()))?;
            if t.dtype() != dtype {
                return Err(format!("{name}: expected {dtype:?}, found {:?}", t.dtype()).into());
            }
            Ok((t.data(), t.shape().to_vec()))
        };
        let halves = |name: &str| -> Result<(Vec<u16>, Vec<usize>)> {
            let (data, shape) = tensor(name, Dtype::F16)?;
            Ok((data.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect(), shape))
        };
        let names = file.names();
        let mut layers = HashMap::new();
        for name in &names {
            let Some(layer) = name.strip_suffix(".a") else { continue };
            let (a, a_shape) = halves(name)?;
            let (db, db_shape) = tensor(&format!("{layer}.db"), Dtype::F32)?;
            let db: Vec<f32> = db.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            // parts `<layer>.b.<first output>`, in the order of their outputs (and of a's rows)
            let prefix = format!("{layer}.b.");
            let mut rows: Vec<usize> = names.iter().filter_map(|n| n.strip_prefix(&prefix)?.parse().ok()).collect();
            rows.sort();
            let parts = rows
                .iter()
                .map(|row| {
                    let (b, shape) = halves(&format!("{prefix}{row}"))?;
                    if shape.len() != 2 {
                        return Err(format!("{prefix}{row}: shape {shape:?}").into());
                    }
                    Ok((*row, b, shape[0]))
                })
                .collect::<Result<Vec<_>>>()?;
            let rank: usize = parts.iter().map(|(_, b, n)| b.len() / n).sum();
            if a_shape.len() != 2 || a_shape[0] != rank || parts.is_empty() || db_shape != [db.len()] {
                return Err(format!("{layer}: adapter of rank {rank}, a {a_shape:?}").into());
            }
            let parts: Vec<(usize, &[u16], usize)> = parts.iter().map(|(r, b, n)| (*r, &b[..], *n)).collect();
            if parts.windows(2).any(|p| p[0].0 + p[0].2 > p[1].0) || parts.last().is_some_and(|p| p.0 + p.2 > db.len())
            {
                return Err(format!("{layer}: parts overlap or pass the layer's {} outputs", db.len()).into());
            }
            layers.insert(layer.to_string(), Arc::new(Lora::new(&a, rank, &parts, &db)));
        }
        if layers.is_empty() {
            return Err(format!("{}: no adapted layers", path.display()).into());
        }
        Ok(Self { layers, metadata })
    }

    pub fn bytes(&self) -> usize {
        self.layers.values().map(|l| l.bytes()).sum()
    }

    /// Attach the adapter to the named layers (`None`: detach every adapter).  Nothing changes unless every
    /// adapted layer exists with the adapter's shape.
    pub fn attach<'a>(
        adapter: Option<&Adapter>,
        layers: impl IntoIterator<Item = (String, &'a mut Linear)>,
    ) -> Result<()> {
        let layers: Vec<(String, &mut Linear)> = layers.into_iter().collect();
        if let Some(a) = adapter {
            for (name, l) in &a.layers {
                let (_, lin) = layers
                    .iter()
                    .find(|(n, _)| n == name)
                    .ok_or_else(|| format!("adapter layer {name}: not in the encoder"))?;
                l.check(name, lin.n, lin.k)?;
            }
        }
        for (name, lin) in layers {
            lin.lora = adapter.and_then(|a| a.layers.get(&name)).cloned();
        }
        Ok(())
    }
}

impl Lora {
    /// That the adapter fits a layer of `n` outputs and `k` inputs.
    pub fn check(&self, name: &str, n: usize, k: usize) -> Result<()> {
        if self.down.k != k || self.n != n {
            return Err(format!("{name}: the adapter does not fit the layer's {n} x {k}").into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm::{self, Act};

    fn leak<T>(v: Vec<T>) -> &'static [T] {
        Box::leak(v.into_boxed_slice())
    }

    #[test]
    fn adapter_adds_its_product_before_the_activation() {
        let (m, n, k, rank) = (19, 40, 70, 5);
        let x: Vec<f32> = (0..m * k).map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 104729) % 97) as f32 / 480.0 - 0.1).collect();
        let a: Vec<f32> = (0..rank * k).map(|i| ((i * 31) % 17) as f32 / 40.0 - 0.2).collect();
        let b: Vec<f32> = (0..n * rank).map(|i| ((i * 13) % 11) as f32 / 30.0 - 0.15).collect();
        let db: Vec<f32> = (0..n).map(|j| j as f32 * 0.01).collect();
        let gain: Vec<f32> = (0..k).map(|p| 1.0 + p as f32 * 0.003).collect();
        let mut lin = Linear {
            n,
            k,
            w: Panels::F32(leak(pack(&w, n, k)).into()),
            scale: None,
            bias: leak(vec![0.1; n]).into(),
            lora: None,
        };
        let half = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect() };
        let (a, b) = (
            half(&a).iter().map(|&h| half::f16::from_bits(h).to_f32()).collect::<Vec<_>>(),
            half(&b).iter().map(|&h| half::f16::from_bits(h).to_f32()).collect::<Vec<_>>(),
        );
        // two parts: outputs 0..24 from a's rows 0..2, outputs 24..40 from rows 2..5
        let (n0, r0) = (24, 2);
        let b0: Vec<f32> = (0..n0).flat_map(|j| b[j * rank..j * rank + r0].to_vec()).collect();
        let b1: Vec<f32> = (n0..n).flat_map(|j| b[j * rank + r0..(j + 1) * rank].to_vec()).collect();
        let parts = [(0, &half(&b0)[..], n0), (n0, &half(&b1)[..], n - n0)];
        lin.lora = Some(Arc::new(Lora::new(&half(&a), rank, &parts, &db)));
        let b: Vec<f32> = (0..n)
            .flat_map(|j| (0..rank).map(move |r| (j, r)))
            .map(|(j, r)| if (j < n0) == (r < r0) { b[j * rank + r] } else { 0.0 })
            .collect();
        let mut out = vec![0.0; m * n];
        gemm::linear(&x, k, m, &lin, &mut out, n, Act::Gelu, false, Some(&gain));
        for i in 0..m {
            let xs: Vec<f64> = (0..k).map(|p| (x[i * k + p] * gain[p]) as f64).collect();
            let low: Vec<f64> = (0..rank).map(|r| (0..k).map(|p| a[r * k + p] as f64 * xs[p]).sum()).collect();
            for j in 0..n {
                let base: f64 = (0..k).map(|p| w[j * k + p] as f64 * xs[p]).sum::<f64>() + 0.1;
                let delta: f64 = (0..rank).map(|r| b[j * rank + r] as f64 * low[r]).sum::<f64>() + db[j] as f64;
                let want = crate::math::gelu((base + delta) as f32) as f64;
                assert!((out[i * n + j] as f64 - want).abs() < 1e-4 * (1.0 + want.abs()), "{i} {j}");
            }
        }
    }
}
