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
mod polygon;
mod trimesh;

pub use extents::*;
pub use grid::*;
pub use line::*;
pub use plane::*;
pub use point::*;
pub use polygon::*;
pub use trimesh::*;

pub trait Envelops<O> {
    fn envelops(&self, object: O) -> bool;
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
