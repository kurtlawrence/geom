//! Common geometry structures and math.
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::{cmp, fmt, iter::FromIterator};

#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

mod extents;
mod grid;
#[cfg(feature = "io")]
pub mod io;
mod line;
mod plane;
mod point;
pub mod polygon;
mod polyline;
mod trimesh;

pub use extents::*;
pub use grid::*;
pub use line::*;
pub use plane::*;
pub use point::*;
pub use polygon::*;
pub use polyline::*;
pub use trimesh::*;

pub trait Envelops<O> {
    fn envelops(&self, object: O) -> bool;
}

/// Area can be calculated from an object.
///
/// Note that area is contextual from the object.
/// For instance, a [`Polygon2`] would be the _plan_ area, a [`TriMesh`] would be the _surface
/// area_, etc.
/// If implementing this trait be sure to be **explicit** about the area being calculated.
pub trait Area {
    /// Calculate the area of an object.
    fn area(&self) -> f64;
}

#[cfg(test)]
fn dummy_grid() -> Grid {
    let mut g = Grid::new([0.0, 0.0], 2, 3, 15.0);
    (0..3)
        .flat_map(|y| (0..2).map(move |x| (x, y)))
        .zip(1..7)
        .for_each(|((x, y), z)| g.set(x, y, z as f64));
    g
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct ExactFloatsGen<const D: usize> {
    pub floats: Vec<f64>,
}

#[cfg(test)]
impl<const D: usize> quickcheck::Arbitrary for ExactFloatsGen<D> {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let floats = std::iter::repeat_with(|| f64::arbitrary(g))
            .take(D)
            .collect();
        Self { floats }
    }
}
