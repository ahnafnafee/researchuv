//! Texel density — the engine's `texel_density` feature (string cluster) plus the
//! addon's `UVPM4_TDensityValue` / `UVPM4_TDensityTierValue` /
//! `UVPM4_TDensityParams` spec: a px-per-unit target applied to islands *before*
//! packing, in tiers.
//!
//! Evidence: engine strings `texel_density`, `find_similar`, `texel_density_set`
//! (set-before-pack), tiered values; addon `spipeline/engine/tdensity.py`
//! (`UVPM4_TDensityUnit` multipliers: meter 1e6, cm 1e4, inch 2540, foot 304.8).

use crate::params::TexelDensityUnit;

/// One texel-density tier — `UVPM4_TDensityTierValue`
/// (a value + the `max_island_dimension` bound it applies under).
#[derive(Clone, Copy, Debug, Default)]
pub struct TexelDensityTier {
    /// Target texel density (px per unit length, already unit-converted).
    pub density: f64,
    /// Applies to islands whose max dimension (pre-scale) is ≤ this (0 = no bound).
    pub max_dim: f64,
}

/// Texel-density parameters — `UVPM4_TDensityParams`.
#[derive(Clone, Debug, Default)]
pub struct TexelDensityParams {
    /// Enable texel density (default false).
    pub enable: bool,
    /// Unit conversion (default px/meter).
    pub unit: TexelDensityUnit,
    /// Target density in the unit's native px/length (default 100).
    pub density: f64,
    /// Tiers (applied when non-empty; overrides the flat density).
    pub tiers: Vec<TexelDensityTier>,
}

/// A texel-density assignment to an island: the scale factor to apply and the
/// resulting px density.
#[derive(Clone, Copy, Debug, Default)]
pub struct TexelDensityPolicy {
    /// Uniform scale factor to apply to the island before packing.
    pub scale: f64,
    /// The effective px density the island is set to (0 = untouched).
    pub density: f64,
}

impl TexelDensityParams {
    /// The target density in absolute px per meter (the display value
    /// converted through the unit: `display / meters_per_display_unit`).
    pub fn density_px_per_meter(&self) -> f64 {
        self.density / self.unit.meters_per_display_unit().max(1e-12)
    }

    /// The tier density in absolute px per meter (tiers share the unit).
    pub fn tier_density_px_per_meter(&self, tier: &TexelDensityTier) -> f64 {
        tier.density / self.unit.meters_per_display_unit().max(1e-12)
    }

    /// The scale factor for an island of the given size (max dimension).
    ///
    /// The engine-side density is 3-D-based (`tex_size · tdensity / scene
    /// unit scale`); this library analog treats the caller's `extent` as the
    /// island's world (meter) size. With a texture of `tex_size` px over a UV
    /// span of `uv_span`, an island covering `extent` m should occupy
    /// `D·extent` px of texture, i.e. a UV footprint of `D·extent·span/tex`
    /// — so the uniform scale is `D·span / (tex·extent)`.
    pub fn scale_for(&self, extent: f64, tex_size: f32, uv_span: f64) -> f64 {
        if !self.enable || extent <= 0.0 {
            return 1.0;
        }
        let density = self.density_px_per_meter();
        let tex = tex_size.max(1.0) as f64;
        let span = if uv_span > 0.0 { uv_span } else { 1.0 };
        (density * span) / (tex * extent)
    }

    /// The policy for an island with a `max_dim` bound check (tiers first).
    pub fn policy_for(&self, max_dim: f64, tex_size: f32, uv_span: f64) -> TexelDensityPolicy {
        if !self.enable {
            return TexelDensityPolicy::default();
        }
        let density = if let Some(tier) = self
            .tiers
            .iter()
            .find(|t| t.max_dim == 0.0 || max_dim <= t.max_dim)
        {
            self.tier_density_px_per_meter(tier)
        } else {
            self.density_px_per_meter()
        };
        if density <= 0.0 || max_dim <= 0.0 {
            return TexelDensityPolicy::default();
        }
        let tex = tex_size.max(1.0) as f64;
        let span = if uv_span > 0.0 { uv_span } else { 1.0 };
        let scale = (density * span) / (tex * max_dim);
        TexelDensityPolicy {
            scale,
            density,
        }
    }
}

/// Set the texel density on a set of island extents (px sizes) — the
/// `texel_density_set` operation: returns per-island scale factors.
pub fn set_tdensity(
    params: &TexelDensityParams,
    extents: &[f64],
    tex_size: f32,
    uv_span: f64,
) -> Vec<TexelDensityPolicy> {
    extents
        .iter()
        .map(|&e| params.policy_for(e, tex_size, uv_span))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_identity() {
        let p = TexelDensityParams::default();
        let out = set_tdensity(&p, &[0.5, 1.0], 1024.0, 1.0);
        assert!(out.iter().all(|o| o.density == 0.0));
    }

    #[test]
    fn px_per_meter_scale() {
        let p = TexelDensityParams {
            enable: true,
            unit: TexelDensityUnit::PxPerMeter,
            density: 100.0, // 100 px/m
            ..TexelDensityParams::default()
        };
        assert!((p.density_px_per_meter() - 100.0).abs() < 1e-9);
        // extent 1 m on a 1024-px texture spanning 1.0 uv: 100 px → 100/1024 uv
        let pol = p.policy_for(1.0, 1024.0, 1.0);
        assert!((pol.scale - 100.0 / 1024.0).abs() < 1e-9);
    }

    #[test]
    fn display_units_convert_to_px_per_meter() {
        // 100 px/cm = 10 000 px/m → 10× the UV footprint of 100 px/m.
        let p = TexelDensityParams {
            enable: true,
            unit: TexelDensityUnit::PxPerCentimeter,
            density: 100.0,
            ..TexelDensityParams::default()
        };
        assert!((p.density_px_per_meter() - 10_000.0).abs() < 1e-6);
        let pol = p.policy_for(1.0, 1024.0, 1.0);
        assert!((pol.scale - 10_000.0 / 1024.0).abs() < 1e-9);
    }

    #[test]
    fn tier_selection() {
        let p = TexelDensityParams {
            enable: true,
            density: 10.0,
            tiers: vec![
                TexelDensityTier {
                    density: 1000.0,
                    max_dim: 0.5,
                },
                TexelDensityTier {
                    density: 100.0,
                    max_dim: 0.0,
                },
            ],
            ..TexelDensityParams::default()
        };
        // Small island → 1000 tier; large island → 100 tier (max_dim 0 = catch-all).
        let small = p.policy_for(0.1, 1024.0, 1.0);
        let big = p.policy_for(2.0, 1024.0, 1.0);
        assert!((small.scale - (1000.0 / 1024.0) / 0.1).abs() < 1e-9);
        assert!((big.scale - (100.0 / 1024.0) / 2.0).abs() < 1e-9);
    }
}
