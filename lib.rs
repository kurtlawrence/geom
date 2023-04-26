//! Data structures.
use rayon::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use rustc_hash::FxHashSet as HashSet;
use std::{cmp, fmt, iter::FromIterator};

pub mod ds;
#[cfg(test)]
mod tests;

// ###### POINTS (VECTORS) ####################################################
/// 2D Point (X,Y).
pub type Point2 = (f64, f64);

/// 3D Point (X,Y,Z).
pub type Point3 = (f64, f64, f64);

pub trait AddPoint<Rhs = Self> {
    fn add(self, rhs: Rhs) -> Self;
    fn sub(self, rhs: Rhs) -> Self
    where
        Self: Sized + Copy,
        Rhs: Point,
    {
        self.add(rhs.scale(-1.0))
    }
}
pub trait Point {
    fn scale(self, scalar: f64) -> Self;
    fn mag(self) -> f64;
    fn unit(self) -> Self
    where
        Self: Sized + Copy,
    {
        self.scale(self.mag().recip())
    }
}

impl AddPoint for Point2 {
    fn add(self, rhs: Self) -> Self {
        (self.0 + rhs.0, self.1 + rhs.1)
    }
}
impl Point for Point2 {
    fn scale(self, scalar: f64) -> Self {
        (self.0 * scalar, self.1 * scalar)
    }
    fn mag(self) -> f64 {
        (self.0 * self.0 + self.1 * self.1).sqrt()
    }
}

impl AddPoint for Point3 {
    fn add(self, rhs: Self) -> Self {
        (self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}
impl Point for Point3 {
    fn scale(self, scalar: f64) -> Self {
        (self.0 * scalar, self.1 * scalar, self.2 * scalar)
    }
    fn mag(self) -> f64 {
        (self.0 * self.0 + self.1 * self.1 + self.2 * self.2).sqrt()
    }
}

impl AddPoint<Point2> for Point3 {
    fn add(self, rhs: Point2) -> Self {
        (self.0 + rhs.0, self.1 + rhs.1, self.2)
    }
}

pub trait ToPoint2 {
    fn to_p2(self) -> Point2;
}

impl ToPoint2 for Point2 {
    fn to_p2(self) -> Point2 {
        self
    }
}
impl ToPoint2 for Point3 {
    fn to_p2(self) -> Point2 {
        (self.0, self.1)
    }
}

pub fn dot_prod(a: Point3, b: Point3) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

#[allow(clippy::many_single_char_names)]
pub fn xprod(a: Point3, b: Point3) -> Point3 {
    let x = a.1 * b.2 - a.2 * b.1;
    let y = a.2 * b.0 - a.0 * b.2;
    let z = a.0 * b.1 - a.1 * b.0;
    (x, y, z)
}

pub fn grade(p3: Point3) -> f64 {
    p3.2 / len_xy(p3)
}

pub fn len_xy(p: impl ToPoint2) -> f64 {
    let p2 = p.to_p2();
    (p2.0 * p2.0 + p2.1 * p2.1).sqrt()
}

pub fn zero_len_xy(p: impl ToPoint2) -> bool {
    let p2 = p.to_p2();
    p2.0.abs() < 1e-7 && p2.1.abs() < 1e-7
}

/// Apply an ordering to points by testing each x,y,z.
pub fn ordpt(a: &Point3, b: &Point3) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if a.0 < b.0 {
        Ordering::Less
    } else if a.0 > b.0 {
        Ordering::Greater
    } else if a.1 < b.1 {
        Ordering::Less
    } else if a.1 > b.1 {
        Ordering::Greater
    } else if a.2 < b.2 {
        Ordering::Less
    } else if a.2 > b.2 {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// Apply a transformation function to each pair of point2s.
#[inline(always)]
fn xfm2(a: Point2, b: Point2, f: impl Fn(f64, f64) -> f64) -> Point2 {
    (f(a.0, b.0), f(a.1, b.1))
}

/// Apply a transformation function to each pair of point3s.
#[inline(always)]
pub fn xfm3(a: Point3, b: Point3, f: impl Fn(f64, f64) -> f64) -> Point3 {
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

// ###### LINE 2D #############################################################
/// A 2D line defined by `Ax + By = C`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Line2 {
    /// The `A` value.
    pub a: f64,
    /// The `B` value.
    pub b: f64,
    /// The `C` value.
    pub c: f64,
}

impl Line2 {
    /// Returns if line is vertical (`B = 0`).
    pub fn is_vertical(&self) -> bool {
        self.b.abs() <= f64::EPSILON
    }

    /// Returns if line is horizontal (`A = 0`).
    pub fn is_horizontal(&self) -> bool {
        self.a.abs() <= f64::EPSILON
    }

    /// The x-intercept, if not horizontal.
    pub fn x_intercept(&self) -> Option<Point2> {
        (!self.is_horizontal())
            .then(|| self.c / self.a)
            .map(|x| (x, 0.0))
    }

    /// The y-intercept, if not vertical.
    pub fn y_intercept(&self) -> Option<Point2> {
        (!self.is_vertical())
            .then(|| self.c / self.b)
            .map(|y| (0.0, y))
    }

    /// Compares a **line against a point** in 2D.
    ///
    /// A _line_ is considered `Less` when it is **beneath** the point, `Equal` when on the
    /// point is on the line,
    /// and `Greater` when **above** the point.
    /// The notion of _beneathness_ is done on the Y axis, for a **vertical** line, _beneath_
    /// is _left_ of the line.
    pub fn pt_cmp(&self, point: Point2) -> cmp::Ordering {
        use cmp::Ordering::*;

        if self.is_vertical() {
            let x1 = self.c / self.a;
            let x2 = point.0;
            let diff = x1 - x2;
            if diff.abs() <= f64::EPSILON {
                Equal // co-linear
            } else if diff > 0.0 {
                Greater // x1 is more than x2, x1 is RIGHT of point
            } else {
                Less // x1 is less than x2, x1 is LEFT of point
            }
        } else {
            let (x, y2) = point;
            let y1 = (self.c - x * self.a) / self.b;
            let diff = y1 - y2;
            if diff.abs() <= f64::EPSILON {
                Equal // co-linear
            } else if diff > 0.0 {
                Greater // y1 is more than y2, y1 is ABOVE point
            } else {
                Less // y1 is less than y2, y1 is BELOW point
            }
        }
    }

    /// Offsets the line _above_ by the value, **perpindicular** to the line.
    /// _Use a negative value to offset the line beneath._
    pub fn offset(&self, d: f64) -> Self {
        // ax + by = c1
        // ax + by = c2
        // d = |c2 - c1| / (a^2 + b^2).sqrt
        // => c2 = c1 + d * (a^2 + b^2).sqrt
        // depending on line parameters it might be +-
        let m = d * (self.a, self.b).mag();
        let c = if self.b < 0.0 || self.is_vertical() && self.a < 0.0 {
            self.c - m
        } else {
            self.c + m
        };

        Self { c, ..*self }
    }
}

impl PartialOrd<Point2> for Line2 {
    fn partial_cmp(&self, point: &Point2) -> Option<cmp::Ordering> {
        Some(self.pt_cmp(*point))
    }
}
impl PartialEq<Point2> for Line2 {
    fn eq(&self, point: &Point2) -> bool {
        self.pt_cmp(*point) == cmp::Ordering::Equal
    }
}

// ###### POLYGON #############################################################
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Polygon2(Vec<Point2>);

impl Polygon2 {
    pub fn new(points: impl Iterator<Item = Point2>) -> Result<Self, &'static str> {
        let points = points.collect::<Vec<_>>();
        if points.len() < 3 {
            Err("polygon requires 3 or more points to be valid")
        } else {
            Ok(Polygon2(points))
        }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn pts(&self) -> &[Point2] {
        &self.0
    }

    /// Test if point is _inside_ the polygon.
    pub fn inside(&self, p: impl ToPoint2) -> bool {
        use std::iter::*;
        // uses the counting winding number test, parallelised!

        /// Test if a point is on a line.
        /// > 0 : p is left of line
        /// = 0 : p is on line
        /// < 0 : p is right of line
        fn on((from, to): (Point2, Point2), p: Point2) -> f64 {
            let f = from.scale(-1.0);
            let a = to.add(f);
            let b = p.add(f);
            a.0 * b.1 - b.0 * a.1
        }

        let point = p.to_p2();
        let one_off = self.len() - 1; // shouldn't underflow since we maintain > 3 pts invariant
        let ps = &self.0;
        let froms = &ps[..one_off];
        let tos = &ps[1..];

        let segments = froms
            .iter()
            .copied()
            .zip(tos.iter().copied())
            .chain(once((ps[one_off], ps[0])));

        let mut winding = 0;

        for (from, to) in segments {
            if from.1 <= point.1 {
                if to.1 > point.1 && on((from, to), point) > 0.0 {
                    winding += 1;
                }
            } else if to.1 <= point.1 && on((from, to), point) < 0.0 {
                winding -= 1;
            }
        }

        winding != 0
    }
}

// ###### GRID ################################################################
/// A grid of generic data.
///
/// A grid is evenly spaced in x and y axis aligned.
///
/// > `PartialEq` is derived and is _exact_ on the float values. This is useful for debugging but
/// should not be used for actual grid equality
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GenericGrid<T> {
    /// The x,y at coord 0,0.
    origin: Point2,

    /// The **X** count, such that the zs are stored row-first.
    stride: usize,

    /// Distance between a coord in one axis.
    spacing: f64,

    /// The z values.
    zs: Vec<Option<T>>,
}

pub struct GridPoint<'a, T> {
    pub grid: &'a GenericGrid<T>,
    pub z: &'a T,
    x: usize,
    y: usize,
}

impl<T> GenericGrid<T> {
    /// Create a new, empty, grid with the given origin, size, and spacing.
    pub fn new(origin: Point2, x_count: usize, y_count: usize, spacing: f64) -> Self {
        let zs = std::iter::repeat_with(|| None)
            .take(x_count * y_count)
            .collect();
        Self {
            origin,
            stride: x_count,
            spacing,
            zs,
        }
    }

    /// The origin the grid was constructed with.
    pub fn origin(&self) -> Point2 {
        self.origin
    }

    /// The number of grid points in the x-axis.
    pub fn x_count(&self) -> usize {
        self.stride
    }

    /// The number of grid points in the y-axis.
    pub fn y_count(&self) -> usize {
        self.zs.len() / self.stride
    }

    /// The spacing between grid points.
    pub fn spacing(&self) -> f64 {
        self.spacing
    }

    /// The _size_ of the grid, this **includes** empty points.
    /// (`x_count * y_count`)
    pub fn len(&self) -> usize {
        self.zs.len()
    }

    /// The number of points that are **not** `None`.
    pub fn len_nonempty(&self) -> usize {
        self.zs.iter().filter(|x| x.is_some()).count()
    }

    /// Returns if the grid is zero-sized, that is, has not points **at all**.
    pub fn is_empty(&self) -> bool {
        self.zs.is_empty()
    }

    /// Returns if the grid has no non-`None` points.
    pub fn is_blank(&self) -> bool {
        self.len_nonempty() == 0
    }

    pub fn get_xy(&self, x: usize, y: usize) -> Point2 {
        let to = (x as f64, y as f64);
        self.origin.add(to.scale(self.spacing))
    }

    pub fn get(&self, x: usize, y: usize) -> Option<GridPoint<T>> {
        self.zs[self.idx(x, y)].as_ref().map(|z| GridPoint {
            grid: self,
            z,
            x,
            y,
        })
    }

    /// Get a mutable point.
    ///
    /// This differs from [`get`] in that it returns just a mutable reference to the z value.
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        let i = self.idx(x, y);
        self.zs[i].as_mut()
    }

    /// Get a point by index rather than location.
    /// **Be sure that idx is taken from a GridPoint**
    pub fn get_idx(&self, idx: usize) -> Option<GridPoint<T>> {
        self.zs[idx].as_ref().map(|z| {
            let y = idx / self.stride;
            let x = idx.saturating_sub(y * self.stride);
            GridPoint {
                grid: self,
                z,
                x,
                y,
            }
        })
    }

    /// Get a mutable point by index rather than location.
    /// **Be sure that idx is taken from a GridPoint.**
    ///
    /// This differs from [`get_idx`] in that it returns just a mutable reference to the z value.
    pub fn get_mut_idx(&mut self, idx: usize) -> Option<&mut T> {
        self.zs.get_mut(idx).and_then(|x| x.as_mut())
    }

    pub fn set(&mut self, x: usize, y: usize, z: impl Into<Option<T>>) {
        let idx = self.idx(x, y);
        self.zs[idx] = z.into();
    }

    /// Get a point by index rather than location.
    /// **Be sure that idx is taken from a GridPoint**
    pub fn set_idx(&mut self, idx: usize, z: impl Into<Option<T>>) {
        self.zs[idx] = z.into();
    }

    /// Take a point, replacing it with `None`.
    pub fn take(&mut self, x: usize, y: usize) -> Option<T> {
        self.take_idx(self.idx(x, y))
    }

    /// Take a point by index rather than location.
    /// **Be sure that idx is taken from a GridPoint**
    pub fn take_idx(&mut self, idx: usize) -> Option<T> {
        std::mem::take(&mut self.zs[idx])
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        if x >= self.stride {
            panic!("x value '{}' is outside grid bounds", x);
        }
        let a = self.stride * y;
        if a >= self.zs.len() {
            panic!("y value '{}' is outside grid bounds", y);
        }

        a + x
    }

    pub fn points(&self) -> impl Iterator<Item = GridPoint<T>> {
        (0..self.y_count())
            .flat_map(move |y| (0..self.x_count()).map(move |x| (x, y)))
            .filter_map(move |(x, y)| self.get(x, y))
    }

    /// Parallel points iterator. The order should not be relied upon.
    pub fn points_par(&self) -> impl rayon::iter::ParallelIterator<Item = GridPoint<T>>
    where
        T: Send + Sync,
    {
        use rayon::prelude::*;
        (0..self.y_count())
            .into_par_iter()
            .flat_map(move |y| (0..self.x_count()).into_par_iter().map(move |x| (x, y)))
            .filter_map(move |(x, y)| self.get(x, y))
    }
}

impl<'a, T> GridPoint<'a, T> {
    pub fn x(&self) -> f64 {
        self.grid.origin().0 + (self.x as f64 * self.grid.spacing())
    }

    pub fn y(&self) -> f64 {
        self.grid.origin().1 + (self.y as f64 * self.grid.spacing())
    }

    pub fn x_idx(&self) -> usize {
        self.x
    }

    pub fn y_idx(&self) -> usize {
        self.y
    }

    pub fn loc(&self) -> (usize, usize) {
        (self.x, self.y)
    }

    /// Index in backing data structure.
    pub fn idx(&self) -> usize {
        self.grid.idx(self.x, self.y)
    }

    pub fn p2(&self) -> Point2 {
        self.grid.get_xy(self.x, self.y)
    }
}

impl<'a> GridPoint<'a, f64> {
    pub fn p3(&self) -> Point3 {
        let (x, y) = self.grid.get_xy(self.x, self.y);
        (x, y, *self.z)
    }
}

impl<'a, T> Clone for GridPoint<'a, T> {
    fn clone(&self) -> Self {
        let Self { grid, z, x, y } = self;
        Self {
            grid,
            z,
            x: *x,
            y: *y,
        }
    }
}
impl<'a, T> Copy for GridPoint<'a, T> {}

impl<T: fmt::Debug> fmt::Debug for GridPoint<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GridPoint: ({x}, {y}) -> ", x = self.x, y = self.y).and_then(|_| self.z.fmt(f))
    }
}

impl<T> PartialEq for GridPoint<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.grid, other.grid) // same grid reference
            && self.x == other.x && self.y == other.y // same x,y position
    }
}

impl From<GridPoint<'_, f64>> for Point3 {
    fn from(g: GridPoint<f64>) -> Self {
        g.p3()
    }
}

impl<T> From<GridPoint<'_, T>> for Point2 {
    fn from(g: GridPoint<T>) -> Self {
        g.p2()
    }
}

impl<T> ToPoint2 for GridPoint<'_, T> {
    fn to_p2(self) -> Point2 {
        self.p2()
    }
}

impl<T> ToPoint2 for &GridPoint<'_, T> {
    fn to_p2(self) -> Point2 {
        self.p2()
    }
}

/// Type alias for a common grid of float values.
pub type Grid = GenericGrid<f64>;

impl Grid {
    /// Sample a [`TriMesh`] on the given `spacing`.
    ///
    /// This effectively _drapes_ a grid over the mesh, using the top or bottom sampled RL.
    pub fn sample(mesh: &TriMesh, spacing: f64, top: bool) -> Self {
        Self::sample_with_bounds(mesh, spacing, top, mesh.aabb())
    }

    /// Sample a [`TriMesh`] on the given `spacing`, within the given `aabb`.
    ///
    /// This effectively _drapes_ a grid over the mesh, using the top or bottom sampled RL.
    /// The returned grid uses `aabb` as the characteristics, so two samples of different meshes
    /// using the same `aabb` can be used to union or intersect grids.
    pub fn sample_with_bounds(
        mesh: &TriMesh,
        spacing: f64,
        top: bool,
        aabb: (Point3, Point3),
    ) -> Self {
        type Map = HashMap<(u32, u32), f64>;

        fn loc_floor(p: Point2, origin: Point2, sp_inv: f64) -> (u32, u32) {
            let diff = p.add(origin.scale(-1.0));
            (
                (diff.0 * sp_inv).floor() as u32,
                (diff.1 * sp_inv).floor() as u32,
            )
        }

        fn sample_tri(
            map: &mut Map,
            tri: Tri,
            origin: Point2,
            spacing: f64,
            sp_inv: f64,
            top: bool,
        ) {
            let plane = Plane::from(&tri);
            if plane.is_vertical() {
                return;
            }

            let (p1, p2, p3) = (tri.0.to_p2(), tri.1.to_p2(), tri.2.to_p2());
            let min = xfm2(xfm2(p1, p2, f64::min), p3, f64::min);
            let max = xfm2(xfm2(p1, p2, f64::max), p3, f64::max);

            let lwr = loc_floor(min, origin, sp_inv);
            let upr = loc_floor(max, origin, sp_inv);

            let poly = Polygon2::new([p1, p2, p3].iter().copied()).expect("given 3 points");

            let pts = (lwr.1..=upr.1).flat_map(|y| (lwr.0..=upr.0).map(move |x| (x, y)));
            for (x, y) in pts {
                let pt = origin.add((x as f64, y as f64).scale(spacing));
                if !poly.inside(pt) {
                    continue;
                }

                let z = plane.register_z(pt);
                add_z(map, (x, y), z, top);
            }
        }

        fn add_z(map: &mut Map, k: (u32, u32), z: f64, top: bool) {
            map.entry(k)
                .and_modify(|a| {
                    if z > *a && top || z < *a && !top {
                        *a = z
                    }
                })
                .or_insert(z);
        }

        let diff = aabb.1.sub(aabb.0);
        let origin = aabb.0.to_p2();
        let sp_inv = spacing.recip();
        // go beyond floor by one spacing
        let xlen = (diff.0 * sp_inv).floor() as usize + 1;
        let ylen = (diff.1 * sp_inv).floor() as usize + 1;

        let map = Map::with_capacity_and_hasher(xlen * ylen, Default::default());

        // do the sampling
        let samples: Map = mesh.tris().fold(map, |mut map, tri| {
            sample_tri(&mut map, tri, origin, spacing, sp_inv, top);
            map
        });

        // apply samples into grid
        let mut grid = Grid::new(origin, xlen, ylen, spacing);
        for ((x, y), z) in samples {
            grid.set(x as usize, y as usize, z);
        }

        grid
    }
}

// ###### TRIMESH #############################################################
/// Triangle represented by 3 points (A, B, C).
/// Boxed since 3 points is 72 bytes.
pub type Tri = Box<(Point3, Point3, Point3)>;

/// A triangle mesh.
///
/// `PartialEq` is _derived_ but does _exact_ equality including structural equality. This is
/// **not** the same as value equality (it is a _subset_ of it) so `PartialEq` should not be used
/// for value equality.
#[derive(Debug, PartialEq, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TriMesh {
    /// The _distinct_ points.
    points: Vec<Point3>,
    /// Each _triangle_ is a triplet of points.
    ///
    /// Each tuple entry is the _index_ back into the `points`.
    /// This is done to save space so a 'triangle' takes up 12 bytes,
    /// and points are not duplicated. (A point itself takes **24 bytes**).
    triangles: Vec<(u32, u32, u32)>,
}

impl TriMesh {
    pub fn point_len(&self) -> usize {
        self.points.len()
    }

    pub fn tri_len(&self) -> usize {
        self.triangles.len()
    }

    pub fn points(&self) -> &[Point3] {
        &self.points
    }

    pub fn tri_indices(&self) -> &[(u32, u32, u32)] {
        &self.triangles
    }

    pub fn tris(&self) -> impl ExactSizeIterator<Item = Tri> + '_ {
        self.triangles
            .iter()
            .map(move |&(a, b, c)| {
                (
                    self.points[a as usize],
                    self.points[b as usize],
                    self.points[c as usize],
                )
            })
            .map(Box::new)
    }

    pub fn extend(&mut self, tris: impl Iterator<Item = Tri>) {
        // call into the extend_trimesh module since there are perf improvements there
        extend_trimesh::extend(self, tris)
    }

    /// Returns a set of polygons that trace the outside of the triangle mesh.
    /// If the mesh is closed this would be an empty vector.
    pub fn outlines(&self) -> Vec<Vec<Point3>> {
        // the outlines algorithm is as follows:
        // 1. get set of _edges_ that only have _one_ tri,
        // 2. join edges together
        // the main algorithm is specified in the module `outline` to enforce some checking and
        // performance code

        // 1. get free edges
        let free_edges = outline::free_edges(self);

        // 2. join edges together
        let outlines = outline::get_outlines(free_edges.into_iter().collect());

        outlines
            .into_iter()
            .map(|o| {
                let take = if o.front() == o.back() {
                    o.len().saturating_sub(1)
                } else {
                    o.len()
                };

                o.into_iter()
                    .take(take)
                    .map(|idx| self.points[idx as usize])
                    .collect()
            })
            .collect()
    }

    /// Returns the axis-align bounding box of the extents of points.
    pub fn aabb(&self) -> (Point3, Point3) {
        // the initial point should generally be valid unless the mesh is empty
        let init = self.points.first().copied().unwrap_or((0.0, 0.0, 0.0));

        self.points.par_iter().map(|&x| (x, x)).reduce(
            || (init, init),
            |(amin, amax), (bmin, bmax)| {
                let min = xfm3(amin, bmin, f64::min);
                let max = xfm3(amax, bmax, f64::max);
                (min, max)
            },
        )
    }

    /// Ensures that there are **not** additional points that do not have faces by removing these
    /// extra points.
    pub fn consolidate(&mut self) {
        let len = self.point_len();
        let mut used = vec![false; len];
        for (a, b, c) in self.triangles.iter().copied() {
            used[a as usize] = true;
            used[b as usize] = true;
            used[c as usize] = true;
        }

        for idx in (0..len).rev().filter(|&x| !used[x]) {
            // remove at idx (reversed so order is maintained)
            // have to _decrement_ all idx references _above_ idx =/
            self.points.remove(idx);
            let idx = idx as u32;
            for x in self.triangles.iter_mut() {
                if x.0 > idx {
                    x.0 -= 1;
                }
                if x.1 > idx {
                    x.1 -= 1;
                }
                if x.2 > idx {
                    x.2 -= 1;
                }
            }
        }
    }
}

impl From<&Grid> for TriMesh {
    fn from(grid: &Grid) -> Self {
        // since we are building from a grid there are some optimisations we can make
        // 1. Grid::points can be collected as Vec<Point3>, at the same time, a map can be made of
        //    (x,y): index
        // 2. Centre points can be added progressively, since they are known to be distinct

        // build points store + lookup
        let len = grid.len();
        let seed = (
            Vec::with_capacity(len * 2), // account for centre points
            HashMap::with_capacity_and_hasher(len, Default::default()),
        );
        let (mut points, pmap) = grid.points().fold(seed, |(mut pts, mut map), p| {
            let idx = pts.len();
            pts.push(p.p3());
            map.insert(p.loc(), idx);
            (pts, map)
        });

        fn centre(a: impl Into<Point3>, b: impl Into<Point3>) -> Point3 {
            a.into().add(b.into()).scale(0.5)
        }

        let i = |p: GridPoint<f64>| *pmap.get(&p.loc()).expect("needs to be in map") as u32;

        // work through each grid cell -- serial
        let mut triangles = Vec::with_capacity(grid.len() * 4);
        let locs = (0..grid.y_count().saturating_sub(1))
            .flat_map(|y| (0..grid.x_count().saturating_sub(1)).map(move |x| (x, y)));
        for (x, y) in locs {
            let ps = (
                grid.get(x, y),
                grid.get(x + 1, y),
                grid.get(x + 1, y + 1),
                grid.get(x, y + 1),
            );

            match ps {
                (Some(p1), Some(p2), Some(p3), Some(p4)) => {
                    // all four points available!
                    let c = p1.p3().add(p2.p3()).add(p3.p3()).add(p4.p3()).scale(0.25);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p1), ci, i(p2))); // bottom
                    triangles.push((i(p2), ci, i(p3))); // right
                    triangles.push((i(p3), ci, i(p4))); // top
                    triangles.push((i(p4), ci, i(p1))); // left
                }
                (None, Some(p2), Some(p3), Some(p4)) => {
                    let c = centre(p2, p4);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p2), ci, i(p3))); // right
                    triangles.push((i(p3), ci, i(p4))); // top
                }
                (Some(p1), None, Some(p3), Some(p4)) => {
                    let c = centre(p1, p3);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p3), ci, i(p4))); // top
                    triangles.push((i(p4), ci, i(p1))); // left
                }
                (Some(p1), Some(p2), None, Some(p4)) => {
                    let c = centre(p2, p4);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p1), ci, i(p2))); // bottom
                    triangles.push((i(p4), ci, i(p1))); // left
                }
                (Some(p1), Some(p2), Some(p3), None) => {
                    let c = centre(p1, p3);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p1), ci, i(p2))); // bottom
                    triangles.push((i(p2), ci, i(p3))); // right
                }
                _ => (),
            }
        }

        triangles.shrink_to_fit();
        points.shrink_to_fit();
        Self { points, triangles }
    }
}

impl FromIterator<Tri> for TriMesh {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Tri>,
    {
        let mut t = TriMesh {
            points: Vec::new(),
            triangles: Vec::new(),
        };
        t.extend(iter.into_iter());
        t
    }
}

mod outline {
    use super::*;
    use std::{collections::VecDeque, hash::*};

    #[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
    pub struct Edge(pub u32, pub u32);

    impl Edge {
        #[inline(always)]
        fn ord(self) -> Self {
            if self.0 <= self.1 {
                self
            } else {
                Edge(self.1, self.0)
            }
        }

        #[inline(always)]
        fn as_u64(&self) -> u64 {
            ((self.0 as u64) << 32) | self.1 as u64
        }
    }
    impl From<(u32, u32)> for Edge {
        #[inline(always)]
        fn from((a, b): (u32, u32)) -> Self {
            Edge(a, b)
        }
    }
    #[allow(clippy::derive_hash_xor_eq)]
    impl Hash for Edge {
        #[inline(always)]
        fn hash<H: Hasher>(&self, hasher: &mut H) {
            hasher.write_u64(self.as_u64())
        }
    }

    pub fn free_edges(mesh: &TriMesh) -> HashSet<Edge> {
        let tris = mesh.triangles.iter().copied();

        let mut set: HashSet<Edge> =
            HashSet::with_capacity_and_hasher(mesh.tri_len(), Default::default());
        let mut ins = |e: Edge| {
            if !set.remove(&e) {
                set.insert(e);
            }
        };

        for (a, b, c) in tris {
            ins(Edge(a, b).ord());
            ins(Edge(a, c).ord());
            ins(Edge(b, c).ord());
        }

        set
    }

    fn join_edges(edges: &mut Vec<Edge>) -> VecDeque<u32> {
        // worst case is O(n²)
        let mut v = VecDeque::with_capacity(edges.len());
        match edges.pop() {
            Some(Edge(a, b)) => {
                v.push_back(a);
                v.push_back(b);
            }
            None => return v,
        }

        // the get algorithm uses swap_remove to linearly search through edges
        let mut get = |pos| {
            let mut r = None;
            // search backwards, since swap_remove swaps in last element
            for i in (0..edges.len()).rev() {
                let Edge(a, b) = edges.swap_remove(i); // last goes to i'th
                if a == pos {
                    r = Some(b); // a matches pos, so b is next point
                } else if b == pos {
                    r = Some(a); // b matches pos, so a is next point
                } else {
                    edges.push(Edge(a, b)); // neither matches, push edge back
                }
                if r.is_some() {
                    break; // found it so break
                }
            }

            r
        };

        let (mut has_front, mut has_back) = (true, true);

        loop {
            if !has_front && !has_back {
                break;
            }

            if has_front {
                let front = *v.front().expect("will have front");
                match get(front) {
                    Some(x) => v.push_front(x),
                    None => has_front = false,
                }
            }

            if has_back {
                let back = *v.back().expect("will have back");
                match get(back) {
                    Some(x) => v.push_back(x),
                    None => has_back = false,
                }
            }
        }

        v
    }

    pub fn get_outlines(free_edges: Vec<Edge>) -> Vec<VecDeque<u32>> {
        let mut edges = free_edges;
        let mut v = Vec::new();

        while !edges.is_empty() {
            let shape = join_edges(&mut edges);
            v.push(shape);
        }

        v
    }
}

mod extend_trimesh {
    use super::*;
    use std::cmp::Ordering;

    const T: f64 = 1e-7;

    // define point wrapper which handles the ordering + equality with tolerance
    struct Pt {
        p: Point3,
        i: u32,
    }
    impl PartialEq for Pt {
        fn eq(&self, rhs: &Self) -> bool {
            (self.p.0 - rhs.p.0).abs() < T
                && (self.p.1 - rhs.p.1).abs() < T
                && (self.p.2 - rhs.p.2).abs() < T
        }
    }
    impl Eq for Pt {}
    impl PartialOrd for Pt {
        fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
            Some(self.cmp(rhs))
        }
    }
    impl Ord for Pt {
        #[allow(clippy::float_cmp)]
        fn cmp(&self, rhs: &Self) -> Ordering {
            if self.eq(rhs) {
                // expensive to check first, but handles the case for equality with tolerance
                return Ordering::Equal;
            }

            let cmp = |a: f64, b: f64| a.partial_cmp(&b).unwrap_or(Ordering::Equal);

            match (self.p.0 == rhs.p.0, self.p.1 == rhs.p.1) {
                (true, true) => cmp(self.p.2, rhs.p.2),
                (true, _) => cmp(self.p.1, rhs.p.1),
                _ => cmp(self.p.0, rhs.p.0),
            }
        }
    }

    pub fn extend(mesh: &mut TriMesh, tris: impl Iterator<Item = Tri>) {
        // first collect the current points->idx into a BTreeSet
        let mut ordset = std::collections::BTreeSet::new();
        ordset.extend(
            mesh.points
                .iter()
                .enumerate()
                .map(|(i, &p)| Pt { p, i: i as u32 }),
        );

        let TriMesh { points, triangles } = mesh;

        let mut get_or_add = |p: Point3| {
            let pt = Pt {
                p,
                i: points.len() as u32,
            };
            match ordset.get(&pt) {
                Some(idx) => idx.i,
                None => {
                    let i = pt.i;
                    points.push(pt.p);
                    ordset.insert(pt);
                    i
                }
            }
        };

        for (a, b, c) in tris.map(|x| *x) {
            let t = (get_or_add(a), get_or_add(b), get_or_add(c));
            triangles.push(t);
        }
    }
}

mod contours {
    use super::*;

    impl TriMesh {
        pub fn contour(&self, spacing: f64) -> Vec<Polyline> {
            use rayon::prelude::*;

            let (min, max) = self.aabb();
            let start = (min.2 / spacing).floor() * spacing;
            let end = (max.2 / spacing).floor() * spacing;
            let steps = ((end - start) / spacing).trunc() as usize;

            let faces = self.faces();

            (0..=steps)
                .into_par_iter()
                .map(|i| {
                    let rl = start + i as f64 * spacing;
                    contour_rl(&faces, rl)
                })
                .reduce(Vec::new, |mut a, b| {
                    a.extend(b.into_iter());
                    a
                })
        }

        pub fn contour_rl(&self, rl: f64) -> Vec<Polyline> {
            let faces = self.faces();
            contour_rl(&faces, rl)
        }

        fn faces(&self) -> Vec<Face> {
            let mut faces = self.tris().map(Face::from).collect::<Vec<_>>();
            faces.sort_unstable_by(|a, b| a.minz.partial_cmp(&b.minz).expect("point isn't a NaN"));
            faces
        }
    }

    type Polyline = Vec<Point3>;
    type Edge3D = (Point3, Point3);
    type Edge = (usize, usize);

    struct Face {
        minz: f64,
        maxz: f64,
        tri: Tri,
    }

    impl From<Tri> for Face {
        fn from(tri: Tri) -> Self {
            let (a, b, c) = (tri.0 .2, tri.1 .2, tri.2 .2);
            let minz = if a < b && a < c {
                a
            } else if b < c {
                b
            } else {
                c
            };
            let maxz = if a > b && a > c {
                a
            } else if b > c {
                b
            } else {
                c
            };
            Self { minz, maxz, tri }
        }
    }

    /// **Assumes `faces` is sorted by minimum z!** Required for search space reduction.
    fn contour_rl(faces: &[Face], rl: f64) -> Vec<Polyline> {
        let gt = faces
            .binary_search_by(|f| f.minz.partial_cmp(&rl).expect("not NaN"))
            .unwrap_or_else(|e| e);
        if gt >= faces.len() {
            return Vec::new();
        }

        // reduce search space (everything > gt does not intersect rl)
        let tris = faces[..=gt].iter().filter(|f| rl <= f.maxz).map(|x| &x.tri);
        let edges = contour_edges(tris, rl);
        let (mut edges, points) = index_edges(edges);
        sort_and_dedup_edges(&mut edges);
        let splits = join_edges(&mut edges);

        let mut start = 0;
        splits
            .into_iter()
            .map(|s| {
                let polyline = flatten_edges(&edges[start..s])
                    .map(|i| points[i])
                    .collect::<Polyline>();
                start = s;
                polyline
            })
            .collect()
    }

    fn contour_edges<'a, I>(tris: I, rl: f64) -> Vec<Edge3D>
    where
        I: Iterator<Item = &'a Tri>,
    {
        tris.filter_map(|t| contour_tri(t, rl)).collect()
    }

    fn contour_tri(tri: &Tri, rl: f64) -> Option<Edge3D> {
        let [s1, s2, d] = split_tri(tri, rl)?;
        let a = point_at_rl(d, s1, rl);
        let b = point_at_rl(d, s2, rl);
        Some((a, b))
    }

    /// Splits the tri into 2 points on one side of the rl, and one point on the other side. The
    /// order/value does not matter, only that two points must be on same side.
    /// Returns `None` if a split does not work (all points on one side), or if all points are on
    /// the same rl.
    /// Returns `Some([same, same, different])`.
    fn split_tri(tri: &Tri, rl: f64) -> Option<[Point3; 3]> {
        const T: f64 = 1e-5;
        let mut arr = [tri.0, tri.1, tri.2];
        if arr.iter().all(|p| (p.2 - rl).abs() < T) {
            return None; // all == rl
        }

        let (a, b, c) = {
            let mut x = arr.iter().map(|p| ((p.2 - rl).signum() - 1.0).abs() < 1e-7);
            (x.next(), x.next(), x.next())
        };

        if a == b && b == c {
            None // all on same side of rl
        } else {
            if a == c {
                arr.swap(1, 2); // move 2nd to 3rd
            } else if b == c {
                arr.swap(0, 2); // swap 1st and last
            }

            #[allow(clippy::float_cmp)]
            {
                debug_assert_eq!((arr[0].2 - rl).signum(), (arr[1].2 - rl).signum());
                debug_assert_ne!((arr[0].2 - rl).signum(), (arr[2].2 - rl).signum());
                debug_assert_ne!((arr[1].2 - rl).signum(), (arr[2].2 - rl).signum());
            }
            Some(arr)
        }
    }

    fn point_at_rl(p1: Point3, p2: Point3, rl: f64) -> Point3 {
        let v = p2.sub(p1);
        let r = (rl - p1.2) / v.2;
        p1.add(v.scale(r))
    }

    /// Each `Edge` has been _ordered_, but the `Vec`tor has **not**.
    fn index_edges(edges: Vec<Edge3D>) -> (Vec<Edge>, Vec<Point3>) {
        let mut pts = Vec::new();
        let mut es = Vec::with_capacity(edges.len());
        for (a, b) in edges {
            let a = get_or_add(&mut pts, a);
            let b = get_or_add(&mut pts, b);

            if a == b {
                continue;
            }

            // order the edges here
            let e = if a < b { (a, b) } else { (b, a) };
            es.push(e);
        }

        (es, pts)
    }

    fn get_or_add(pts: &mut Vec<Point3>, p: Point3) -> usize {
        match pts
            .iter()
            .enumerate()
            .find(|x| same_point(&p, x.1))
            .map(|x| x.0)
        {
            Some(idx) => idx,
            None => {
                let idx = pts.len();
                pts.push(p);
                idx
            }
        }
    }

    fn same_point(a: &Point3, b: &Point3) -> bool {
        const T: f64 = 1e-7;
        (a.0 - b.0).abs() < T && (a.1 - b.1).abs() < T && (a.2 - b.2).abs() < T
    }

    fn sort_and_dedup_edges(edges: &mut Vec<Edge>) {
        edges.sort_unstable();
        edges.dedup();
    }

    /// Orders `edges` such that `e0.1 == e1.0`
    /// If edges do not join, this index defines the 'split' which dictates a _new polyline_.
    /// Edges are 'flipped' if matched on opposite orientation.
    fn join_edges(edges: &mut [Edge]) -> Vec<usize> {
        let mut is = Vec::new();
        let mut slice = edges;
        let mut offset = 0;

        while !slice.is_empty() {
            let i = join_edges_slice(slice);
            offset += i;
            is.push(offset);
            slice = &mut slice[i..];
        }

        is
    }

    fn join_edges_slice(edges: &mut [Edge]) -> usize {
        fn rev(edges: &mut [Edge]) {
            edges.reverse();
            for e in edges {
                *e = (e.1, e.0); // flip the edges around
            }
        }

        if edges.is_empty() {
            panic!("called with empty slice");
        }

        let len = edges.len();
        for i in 0..len {
            let mut front = edges[0].0;
            let mut back = edges[i].1;
            let mut found = false;
            let swp = i + 1;

            for j in swp..len {
                let (a, b) = edges[j];

                if a == front || b == front {
                    // this edge matches the _front_. We need to rev and flip front/back
                    rev(&mut edges[..swp]); // reverses up to (not incl) j
                    std::mem::swap(&mut front, &mut back);
                    // double check that this rev/swap does what I expect it to
                    debug_assert_eq!(front, edges[0].0);
                    debug_assert_eq!(back, edges[i].1);
                }

                if a == back {
                    // back.1 == e.0, which is the best case!
                    // break the inner loop and let it progress
                    edges.swap(swp, j);
                    found = true;
                    break;
                } else if b == back {
                    // back.1 == e.1, need to flip e around
                    // break the inner loop and let it progress
                    edges[j] = (b, a);
                    edges.swap(swp, j);
                    found = true;
                    break;
                }
            }

            if !found {
                return swp;
            }
            if edges[swp].1 == front {
                return swp + 1;
            }
        }

        len
    }

    fn flatten_edges(edges: &[Edge]) -> impl Iterator<Item = usize> + '_ {
        edges
            .get(0)
            .map(|x| x.0)
            .into_iter()
            .chain(edges.iter().map(|x| x.1))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn join_edges_slice_2() {
            let mut edges = [(0, 1), (1, 2)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(0, 1), (1, 2)]);

            let mut edges = [(0, 1), (2, 0)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(1, 0), (0, 2)]);

            let mut edges = [(0, 1), (2, 1)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(0, 1), (1, 2)]);
        }

        #[test]
        fn join_edges_slice_3() {
            let mut edges = [(0, 1), (1, 2), (2, 3)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 3);
            assert_eq!(edges, [(0, 1), (1, 2), (2, 3)]);

            let mut edges = [(0, 1), (2, 0), (2, 3)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 3);
            assert_eq!(edges, [(1, 0), (0, 2), (2, 3)]);

            let mut edges = [(0, 1), (3, 4), (2, 1)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(0, 1), (1, 2), (3, 4)]);
        }

        #[test]
        fn join_edges_slice_4() {
            let mut edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 4);
            assert_eq!(edges, [(3, 2), (2, 1), (1, 0), (0, 3)]);

            let mut edges = [(0, 1), (2, 0), (2, 3), (1, 4)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 4);
            assert_eq!(edges, [(3, 2), (2, 0), (0, 1), (1, 4)]);

            let mut edges = [(0, 1), (3, 4), (2, 1), (5, 6)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(0, 1), (1, 2), (3, 4), (5, 6)]);

            let mut edges = [(0, 1), (3, 4), (0, 2), (1, 2)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 3);
            assert_eq!(edges, [(2, 0), (0, 1), (1, 2), (3, 4)]);
        }

        #[test]
        fn join_edges_01() {
            let mut edges = [(0, 1), (2, 1)];
            let v = join_edges(&mut edges);
            assert_eq!(v, vec![2]);
            assert_eq!(edges, [(0, 1), (1, 2)]);
        }

        #[test]
        fn join_edges_02() {
            let mut edges = [(0, 1), (2, 1), (3, 4)];
            let v = join_edges(&mut edges);
            assert_eq!(v, vec![2, 3]);
            assert_eq!(edges, [(0, 1), (1, 2), (3, 4)]);
        }

        #[test]
        fn join_edges_03() {
            let mut edges = [(0, 1), (3, 4), (2, 1), (5, 6)];
            let v = join_edges(&mut edges);
            assert_eq!(v, vec![2, 3, 4]);
            assert_eq!(edges, [(0, 1), (1, 2), (3, 4), (5, 6)]);

            let mut edges = [(0, 1), (3, 4), (0, 2), (1, 2)];
            let v = join_edges(&mut edges);
            assert_eq!(v, vec![3, 4]);
            assert_eq!(edges, [(2, 0), (0, 1), (1, 2), (3, 4)]);
        }

        fn dummy_tri() -> TriMesh {
            let mut g = Grid::new((1.0, 2.0), 4, 3, 15.0);
            for (z, (x, y)) in (0..4).flat_map(|x| (0..3).map(move |y| (x, y))).enumerate() {
                g.set(x, y, z as f64);
            }

            TriMesh::from(&g)
        }

        #[test]
        fn contour_rl_test() {
            let t = &dummy_tri();
            let c = t.contour_rl(-1.0);
            assert!(c.is_empty());

            let c = t.contour_rl(5.0);
            assert_eq!(
                &c,
                &[vec![
                    (16.0, 32.0, 5.0),
                    (19.75, 20.75, 5.0),
                    (21.0, 17.0, 5.0),
                    (23.5, 9.5, 5.0),
                    (26.0, 2.0, 5.0)
                ]]
            );
        }

        #[test]
        fn split_tri_test() {
            let t = Box::new(((0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 1.0, 1.0)));

            let x = split_tri(&t.clone(), -1.0);
            assert_eq!(x, None);
            let x = split_tri(&t.clone(), 2.0);
            assert_eq!(x, None);
            let x = split_tri(&t.clone(), 0.0);
            assert_eq!(x, None);
            let x = split_tri(&t.clone(), 1.0);
            assert_eq!(x, Some([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 1.0, 1.0)]));
            let x = split_tri(&t.clone(), 0.7);
            assert_eq!(x, Some([(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 1.0, 1.0)]));
            let x = split_tri(&t.clone(), 0.2);
            assert_eq!(x, Some([(0.0, 1.0, 1.0), (0.0, 0.0, 0.5), (0.0, 0.0, 0.0)]));
            let x = split_tri(&t.clone(), 0.5);
            assert_eq!(x, Some([(0.0, 1.0, 1.0), (0.0, 0.0, 0.5), (0.0, 0.0, 0.0)]));

            let t = Box::new(((0.0, 0.0, 0.5), (0.0, 0.0, 0.5), (0.0, 1.0, 0.5)));

            let x = split_tri(&t.clone(), 0.5);
            assert_eq!(x, None);
        }

        #[test]
        fn split_tri_test2() {
            let t = Box::new(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.5)));
            let x = split_tri(&t.clone(), 0.7);
            assert_eq!(x, Some([(0.0, 0.0, 0.0), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0)]));
        }
        #[test]
        fn point_at_rl_test() {
            let p = point_at_rl((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.5);
            assert_eq!(p, (0.0, 0.0, 0.5));

            let p = point_at_rl((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), 0.5);
            assert_eq!(p, (5.0, 5.0, 0.5));

            let p = point_at_rl((10.0, 10.0, 1.0), (0.0, 0.0, 0.0), 0.5);
            assert_eq!(p, (5.0, 5.0, 0.5));

            let p = point_at_rl((10.0, 10.0, 1.0), (0.0, 0.0, 0.0), 0.8);
            assert_eq!(p, (8.0, 8.0, 0.8));

            let p = point_at_rl((10.0, 10.0, 0.0), (0.0, 0.0, 1.0), 0.8);
            assert_eq!(p, (2.0, 2.0, 0.8));
        }

        #[test]
        fn contour_tri_test() {
            let t = Box::new(((0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0)));
            let x = contour_tri(&t, 0.5);
            assert_eq!(x, Some(((0.5, 0.5, 0.5), (0.0, 0.5, 0.5))));
        }

        #[test]
        fn flatten_edges_test() {
            let x = flatten_edges(&[(1, 2), (2, 4), (4, 0), (0, 1)]).collect::<Vec<_>>();
            assert_eq!(&x, &[1, 2, 4, 0, 1]);

            let x = flatten_edges(&[(1, 2)]).collect::<Vec<_>>();
            assert_eq!(&x, &[1, 2]);
        }
    }
}

// ###### PLANE ###############################################################
/// Ax + By + Cz = D
#[derive(Debug, Clone)]
pub struct Plane {
    d: f64,
    normal: Point3,
    centroid: Point3,
}

impl Plane {
    pub fn new(centroid: Point3, normal: Point3) -> Self {
        let d = normal.0 * centroid.0 + normal.1 * centroid.1 + normal.2 * centroid.2;
        Self {
            d,
            normal,
            centroid,
        }
    }

    /// Fits a plane using the least squares method.
    /// Returns `None` if less than three points are used.
    ///
    /// Algorithm from <https://www.ilikebigbits.com/2015_03_04_plane_from_points.html>
    #[allow(clippy::suspicious_operation_groupings)]
    #[allow(clippy::float_cmp)]
    pub fn fit_least_sqs(points: &[Point3]) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }

        // Calculate centroid
        let centroid = points
            .iter()
            .copied()
            .fold(Point3::default(), Point3::add)
            .scale((points.len() as f64).recip());

        // Calculate full 3x3 covariance matrix, excluding symmetries
        let n_centroid = centroid.scale(-1.0); // negative centroid.
        let [mut xx, mut xy, mut xz, mut yy, mut yz, mut zz] = [0f64; 6];
        for p in points {
            let r = p.add(n_centroid); // subtract centroid from point
            xx += r.0 * r.0; // xx
            xy += r.0 * r.1; // xy
            xz += r.0 * r.2; // xz
            yy += r.1 * r.1; // yy
            yz += r.1 * r.2; // yz
            zz += r.2 * r.2; // zz
        }

        // Determinants
        let (detx, dety, detz) = (yy * zz - yz * yz, xx * zz - xz * xz, xx * yy - xy * xy);
        let det_max = if detx > dety && detx > detz {
            detx
        } else if dety > detz {
            dety
        } else {
            detz
        };
        if det_max <= 0.0 {
            return None;
        }

        // Pick path with best conditioning
        let dir = if det_max == detx {
            (detx, xz * yz - xy * zz, xy * yz - xz * yy)
        } else if det_max == dety {
            (xz * yz - xy * zz, dety, xy * xz - yz * xx)
        } else {
            (xy * yz - xz * yy, xy * xz - yz * xx, detz)
        };

        Some(Self::new(centroid, dir))
    }

    pub fn a(&self) -> f64 {
        self.normal.0
    }

    pub fn b(&self) -> f64 {
        self.normal.1
    }

    pub fn c(&self) -> f64 {
        self.normal.2
    }

    pub fn d(&self) -> f64 {
        self.d
    }

    pub fn centroid(&self) -> Point3 {
        self.centroid
    }

    pub fn normal(&self) -> Point3 {
        self.normal
    }

    pub fn strike(&self) -> Point3 {
        (self.b(), self.a() * -1.0, 0.0)
    }

    pub fn dip(&self) -> Point3 {
        xprod(self.normal, self.strike())
    }

    pub fn is_vertical(&self) -> bool {
        self.c() == 0.0
    }

    /// # Panics
    /// Panics is plane is vertical (c == 0).
    pub fn register_z(&self, p: impl ToPoint2) -> f64 {
        if self.c() == 0.0 {
            panic!("plane is vertical");
        }

        let p = p.to_p2();

        let i = self.d() - self.a() * p.0 - self.b() * p.1;
        i / self.c()
    }

    /// Returns the **point on the plane** that is the shortest distance from `p`.
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::suspicious_operation_groupings)]
    pub fn shortest_dist(&self, (x, y, z): Point3) -> Point3 {
        // https://math.stackexchange.com/questions/2758190/shortest-distance-from-point-to-a-plane
        // finding point b such that p + tN = b (point on plane)
        let (a, b, c) = self.normal;
        let d = self.d;

        let t = (d - a * x - b * y - c * z) / (a * a + b * b + c * c);

        (x, y, z).add(self.normal.scale(t)) // p + tN
    }
}

impl From<&Tri> for Plane {
    fn from(tri: &Tri) -> Self {
        let n = tri.0.scale(-1.0);
        let a = tri.1.add(n); // p1 - p0
        let b = tri.2.add(n); // p2 - p0
        Plane::new(tri.0, xprod(a, b))
    }
}
