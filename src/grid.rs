use crate::*;

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

    /// The 2D plan extents that this grid can cover.
    ///
    /// This does not account for grid point existence, just the origin, the spacing, and
    /// the lengths.
    pub fn extents(&self) -> Extents2 {
        let size = [self.x_count(), self.y_count()].map(|x| x as f64 * self.spacing);
        Extents {
            origin: self.origin,
            size,
        }
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

    /// Returns if the grid is zero-sized, that is, has no points **at all**.
    pub fn is_empty(&self) -> bool {
        self.zs.is_empty()
    }

    /// Returns if the grid has no non-`None` points.
    pub fn is_blank(&self) -> bool {
        self.len_nonempty() == 0
    }

    /// Returns the [`Point2`] coordinates of the `x` and `y` indices.
    ///
    /// Note that this will always return a value, even if a grid point does not exist.
    pub fn get_xy(&self, x: usize, y: usize) -> Point2 {
        let to = [x as f64, y as f64];
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

    pub fn map<F: FnMut(T) -> U, U>(self, mut f: F) -> GenericGrid<U> {
        let Self {
            origin,
            stride,
            spacing,
            zs,
        } = self;
        let zs = zs.into_iter().map(|x| x.map(&mut f)).collect();

        GenericGrid {
            origin,
            stride,
            spacing,
            zs,
        }
    }

    /// Downsample the grid by _dropping_ `skip` points between points.
    ///
    /// Note that a `skip` of 0 just returns the grid.
    pub fn downsample(self, skip: usize) -> Self {
        if skip == 0 {
            return self;
        }

        let Self {
            origin,
            stride,
            spacing,
            mut zs,
        } = self;

        // to efficiently drop the points, we can use .retain on the backing vector
        // retain visits elements in order exactly once, so we can keep an index track
        // to work out if it needs dropping or not
        // for example, in a 3x3 grid, we drop the 1,3,4,5,7 indices.
        // that is, x=1 || y=1
        // That is if skip is 1.
        // When testing for larger skips, it then becomes a case of checking the mod of skip + 1
        // (7x7 skip 2):
        // x0 keep (0 % 3 == 0)
        // x1 skip
        // x2 skip
        // x3 keep (3 % 3 == 0)
        // x4 skip
        // x5 skip
        // x6 keep (6 % 6 == 0)
        let mut idx = 0;
        let modby = skip + 1;
        zs.retain(|_| {
            let i = idx;
            idx += 1;

            let y = i / stride;
            if y % modby != 0 {
                return false;
            }

            let x = i.saturating_sub(y * stride);
            x % modby == 0
        });

        // the stride reduces by the floor of (skip + 1)
        let stride = (stride + skip) / modby;

        // the spacing expands by the (skip + 1)
        let spacing = spacing * modby as f64;

        Self {
            zs,
            origin, // origin does not change
            stride,
            spacing,
        }
    }

    /// Consume the grid, returning the backing data array.
    pub fn into_zs(self) -> Vec<Option<T>> {
        self.zs
    }
}

impl<'a, T> GridPoint<'a, T> {
    pub fn x(&self) -> f64 {
        self.grid.origin()[0] + (self.x as f64 * self.grid.spacing())
    }

    pub fn y(&self) -> f64 {
        self.grid.origin()[1] + (self.y as f64 * self.grid.spacing())
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
        let [x, y] = self.grid.get_xy(self.x, self.y);
        [x, y, *self.z]
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
        Self::sample_with_bounds(mesh.tris(), spacing, top, mesh.aabb().into())
    }

    /// Sample a bunch of [`Tri`]s on the given `spacing`, within the given `aabb`.
    ///
    /// This effectively _drapes_ a grid over the mesh, using the top or bottom sampled RL.
    /// The returned grid uses `aabb` as the characteristics, so two samples of different meshes
    /// using the same `aabb` can be used to union or intersect grids.
    pub fn sample_with_bounds<T>(tris: T, spacing: f64, top: bool, aabb: Extents2) -> Self
    where
        T: IntoIterator<Item = Tri>,
    {
        type Map = HashMap<(u32, u32), f64>;

        fn loc_floor(p: Point2, origin: Point2, sp_inv: f64) -> (u32, u32) {
            let [x, y] = p.add(origin.scale(-1.0)).scale(sp_inv);
            (x.floor() as u32, y.floor() as u32)
        }

        fn sample_tri(
            map: &mut Map,
            tri: Tri,
            origin: Point2,
            spacing: f64,
            sp_inv: f64,
            top: bool,
        ) {
            let plane = Plane::from(tri);
            if plane.is_vertical() {
                return;
            }

            let [p1, p2, p3] = tri.map(ToPoint2::to_p2);
            let min = p1.min_all(p2).min_all(p3);
            let max = p1.max_all(p2).max_all(p3);

            let lwr = loc_floor(min, origin, sp_inv);
            let upr = loc_floor(max, origin, sp_inv);

            let pinside = |p| polygon::point_inside(&[p1, p2, p3], p);

            let pts = (lwr.1..=upr.1).flat_map(|y| (lwr.0..=upr.0).map(move |x| (x, y)));
            for (x, y) in pts {
                let pt = origin.add([x as f64, y as f64].scale(spacing));
                if !pinside(pt) {
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

        let origin = aabb.origin.to_p2();
        let sp_inv = spacing.recip();
        let [xlen, ylen] = aabb.size.to_p2().scale(sp_inv);
        // go beyond floor by one spacing
        let xlen = xlen.floor() as usize + 1;
        let ylen = ylen.floor() as usize + 1;

        let map = Map::with_capacity_and_hasher(xlen * ylen, Default::default());

        // do the sampling
        let samples: Map = tris.into_iter().fold(map, |mut map, tri| {
            sample_tri(&mut map, tri, origin, spacing, sp_inv, top);
            map
        });

        // apply samples into grid
        let mut grid = Grid::new(origin, xlen, ylen, spacing);
        samples
            .into_iter()
            // if a triangle extends past the aabb bounds, it will have indices
            // that extend past the grid, so we filter these out.
            .filter(|((x, y), _)| *x < xlen as u32 && *y < ylen as u32)
            .for_each(|((x, y), z)| grid.set(x as usize, y as usize, z));

        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_grid() {
        let g = Grid::new([0.0, 0.0], 10, 20, 15.0);
        assert_eq!(g.origin(), [0.0, 0.0]);
        assert_eq!(g.x_count(), 10);
        assert_eq!(g.y_count(), 20);
        assert_eq!(g.spacing(), 15.0);
        assert_eq!(g.len(), 200);
    }

    #[test]
    fn get_xy_test() {
        let g = Grid::new([0.0, 0.0], 10, 20, 15.0);
        assert_eq!(g.get_xy(0, 0), [0.0, 0.0]);
        assert_eq!(g.get_xy(1, 0), [15.0, 0.0]);
        assert_eq!(g.get_xy(0, 1), [0.0, 15.0]);
        assert_eq!(g.get_xy(3, 4), [45.0, 60.0]);
    }

    #[test]
    fn populate_grid() {
        let g = dummy_grid();

        assert_eq!(
            &g.zs,
            &[
                Some(1.0),
                Some(2.0),
                Some(3.0),
                Some(4.0),
                Some(5.0),
                Some(6.0)
            ]
        );

        assert_eq!(g.get(0, 0).map(|p| p.p3()), Some([0.0, 0.0, 1.0]));
        assert_eq!(g.get(1, 0).map(|p| p.p3()), Some([15.0, 0.0, 2.0]));
        assert_eq!(g.get(0, 1).map(|p| p.p3()), Some([0.0, 15.0, 3.0]));
        assert_eq!(g.get(1, 1).map(|p| p.p3()), Some([15.0, 15.0, 4.0]));
        assert_eq!(g.get(0, 2).map(|p| p.p3()), Some([0.0, 30.0, 5.0]));
        assert_eq!(g.get(1, 2).map(|p| p.p3()), Some([15.0, 30.0, 6.0]));
    }

    #[test]
    fn grid_points() {
        let g = dummy_grid();
        let mut points = g.points();

        let p = points.next().unwrap();
        assert_eq!(p.x(), 0.0);
        assert_eq!(p.x_idx(), 0);
        assert_eq!(p.y(), 0.0);
        assert_eq!(p.y_idx(), 0);
        assert_eq!(p.p2(), [0.0, 0.0]);
        assert_eq!(p.p3(), [0.0, 0.0, 1.0]);

        assert_eq!(
            points.next().map(|p| (p.idx(), p.p3())),
            Some((1, [15.0, 0.0, 2.0]))
        );
        assert_eq!(
            points.next().map(|p| (p.idx(), p.p3())),
            Some((2, [0.0, 15.0, 3.0]))
        );
        assert_eq!(points.next().map(|p| p.p3()), Some([15.0, 15.0, 4.0]));
        assert_eq!(points.next().map(|p| p.p3()), Some([0.0, 30.0, 5.0]));
        assert_eq!(
            points.next().map(|p| (p.idx(), p.p3())),
            Some((5, [15.0, 30.0, 6.0]))
        );
    }

    #[test]
    fn grid_downsample_smoke() {
        let origin = Point2::zero();
        let g = Grid::new(origin, 2, 2, 2.0);
        let g2 = g.clone().downsample(0);
        assert_eq!(g, g2);

        // 2x2 skip 1: expecting to drop just to a single point
        let g = Grid::new(origin, 2, 2, 2.0).downsample(1);
        assert_eq!(g.len(), 1);
        assert_eq!(g.x_count(), 1);
        assert_eq!(g.y_count(), 1);
        assert_eq!(g.spacing(), 4.0);

        // 3x3 skip 1
        let g = Grid::new(origin, 3, 3, 2.0).downsample(1);
        assert_eq!(g.len(), 4);
        assert_eq!(g.x_count(), 2);
        assert_eq!(g.y_count(), 2);
        assert_eq!(g.spacing(), 4.0);

        // 3x5 skip 1
        let g = Grid::new(origin, 3, 5, 2.0).downsample(1);
        assert_eq!(g.len(), 6);
        assert_eq!(g.x_count(), 2);
        assert_eq!(g.y_count(), 3);
        assert_eq!(g.spacing(), 4.0);

        // 6x6 skip 2
        let g = Grid::new(origin, 6, 6, 2.0).downsample(2);
        assert_eq!(g.len(), 4);
        assert_eq!(g.x_count(), 2);
        assert_eq!(g.y_count(), 2);
        assert_eq!(g.spacing(), 6.0);

        // 7x7 skip 2
        let g = Grid::new(origin, 7, 7, 2.0).downsample(2);
        assert_eq!(g.len(), 9);
        assert_eq!(g.x_count(), 3);
        assert_eq!(g.y_count(), 3);
        assert_eq!(g.spacing(), 6.0);
    }
}
