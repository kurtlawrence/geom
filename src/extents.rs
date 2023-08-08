use crate::*;

pub type Extents2 = Extents<Point2>;
pub type Extents3 = Extents<Point3>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Extents<P> {
    pub origin: P,
    pub size: P,
}

impl<P> Extents<P>
where
    P: Copy + Point + Add,
{
    pub fn zero() -> Self {
        Self {
            origin: P::zero(),
            size: P::zero(),
        }
    }

    pub fn from_min_max(min: P, max: P) -> Self {
        let size = max.sub(min);

        Self { origin: min, size }
    }

    pub fn max(&self) -> P {
        self.origin.add(self.size)
    }

    pub fn union(self, other: Self) -> Self {
        let origin = self.origin.min_all(other.origin);
        let max = self.max().max_all(other.max());
        let size = max.sub(origin);

        Self { origin, size }
    }

    pub fn intersection(self, other: Self) -> Option<Self>
    where
        Self: Envelops<P>,
    {
        let origin = self.origin.max_all(other.origin);
        let max = self.max().min_all(other.max());

        // ensure origin and max are both inside the original extents
        (self.envelops(origin)
            && self.envelops(max)
            && other.envelops(origin)
            && other.envelops(max))
        .then(|| Self {
            origin,
            size: max.sub(origin),
        })
    }

    pub fn intersects(self, other: Self) -> bool {
        let outside = self.max().into_iter().zip(other.origin).any(|(m, c)| m < c)
            || self.origin.into_iter().zip(other.max()).any(|(o, c)| o > c);

        !outside
    }

    /// Expand the extents by a value.
    ///
    /// A negative value can be used to _shrink_ the extents.
    /// Note that shrinking beyond `size / 2` will result in a [`Self::zero`].
    /// `by` values that a non-finite are ignored and the original self is returned.
    ///
    /// # Example
    /// ```rust
    /// # use geom::*;
    /// let e = Extents2::from_min_max(Point2::zero(), Point2::one());
    ///
    /// assert_eq!(e.expand(0.5), Extents2 {
    ///     origin: [-0.5, -0.5],
    ///     size: [2.0, 2.0]
    /// });
    ///
    /// assert_eq!(e.expand(-0.2), Extents2 {
    ///     origin: [0.2, 0.2],
    ///     size: [0.6, 0.6],
    /// });
    /// ```
    pub fn expand(self, by: f64) -> Self {
        if !by.is_finite() {
            return self;
        }

        let x = P::all(by);

        let size = self.size.add(x.scale(2.0));
        if size.into_iter().any(|x| x < 0.0) {
            return Self::zero();
        }

        let origin = self.origin.sub(x);

        Self { origin, size }
    }
}

impl Extents3 {
    /// A three-dimensional AABB has volume!
    pub fn volume(&self) -> f64 {
        let [w, d, h] = self.size;
        w * d * h
    }

    /// Return the 8 corners of this box.
    pub fn corners(&self) -> [Point3; 8] {
        let [x0, y0, z0] = self.origin;
        let [x1, y1, z1] = self.max();
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ]
    }

    /// Cut the box with a plane, returning the intersection points.
    ///
    /// Cutting a bounding box is useful for _reducing_ an AABB by a plane (such as a view plane).
    /// This function will only return the points that intersect a cutting plane, what side to keep
    /// should be tested with [`Plane::lies`]. An empty vector indicates that the AABB lies
    /// completely on one side of the plane.
    ///
    /// > **Note that _coincident_ points are _excluded_ as intersection points.**
    /// > This is done because a 'cut' is intended to partition the AABB's corners into categories,
    /// > so including coincident points would lead to ambiguity.
    ///
    /// # Examples
    ///
    /// ## Reducing an AABB with a view plane
    /// ```rust
    /// # use geom::*;
    /// let aabb = Extents3::from_min_max(Point3::zero(), Point3::one());
    /// let plane = Plane::from([
    ///     [0.5, 0.0, 0.0],
    ///     [0.0, 0.5, 0.0],
    ///     [0.0, 0.0, 0.5],
    /// ]);
    ///
    /// let x = aabb.cut(&plane, 1e-7);
    /// assert_eq!(x.len(), 3);
    /// assert_eq!(x, &[
    ///     [0.5, 0.0, 0.0],
    ///     [0.0, 0.5, 0.0],
    ///     [0.0, 0.0, 0.5],
    /// ]);
    ///
    /// let reduced_aabb = Extents3::from_iter(
    ///     x.into_iter().chain(std::iter::once(aabb.origin))
    /// );
    /// assert_eq!(reduced_aabb.max(), [0.5, 0.5, 0.5]);
    /// ```
    ///
    /// ## Coincident points are not intersection points
    /// ```rust
    /// # use geom::*;
    /// let aabb = Extents3::from_min_max(Point3::zero(), Point3::one());
    /// let plane = Plane::from([
    ///     [0.5, 0.0, 0.0],
    ///     [0.5, 0.5, 0.0],
    ///     [1.0, 1.0, 1.0],
    /// ]);
    ///
    /// let x = aabb.cut(&plane, 1e-7);
    /// assert_eq!(x.len(), 2);
    /// assert_eq!(x, &[
    ///     [0.5, 0.0, 0.0],
    ///     [0.5, 1.0, 0.0],
    /// ]);
    /// ```
    ///
    /// ## Extents lies on one side of plane
    /// ```rust
    /// # use geom::*;
    /// let aabb = Extents3::from_min_max(Point3::zero(), Point3::one());
    /// // notice that it conincides with aabb
    /// let plane = Plane::from([
    ///     [2.0, 0.0, 0.0],
    ///     [1.0, 0.0, 1.0],
    ///     [1.0, 1.0, 1.0],
    /// ]);
    ///
    /// let x = aabb.cut(&plane, 1e-7);
    /// assert!(x.is_empty());
    /// ```
    pub fn cut(&self, plane: &Plane, tolerance: f64) -> Vec<Point3> {
        let corners = self.corners();
        let corners_ = corners.map(|p| plane.lies(p, tolerance));

        let segments = [
            // front
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            // connectors
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            // back
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
        ];

        let mut v = Vec::new();

        for seg in segments {
            // there are only two cases that we want to find the intersection and that
            // happens when the segment a/b has inverted Lies values.
            let [a, b] = seg.map(|x| corners_[x]);
            if a == Lies::On || a != b.inv() {
                continue;
            }

            let [a, b] = seg.map(|x| corners[x]);

            // line-plane intersection
            let l = b.sub(a);
            let num = dot_prod(plane.centroid().sub(a), plane.normal());
            let den = dot_prod(l, plane.normal());
            let d = num / den;

            let p = a.add(l.scale(d));
            v.push(p);
        }

        return v;
    }
}

impl From<Extents3> for Extents2 {
    /// Convert a 3D AABB to 2D AABB by dropping Z value.
    fn from(value: Extents3) -> Self {
        let Extents { origin, size } = value;
        Self {
            origin: origin.to_p2(),
            size: size.to_p2(),
        }
    }
}

impl FromIterator<Point3> for Extents3 {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Point3>,
    {
        let mut iter = iter.into_iter();
        let Some(init) = iter.next() else { return Self::zero(); };

        let (min, max) = iter.fold((init, init), |(min, max), p| {
            (min.min_all(p), max.max_all(p))
        });

        Self::from_min_max(min, max)
    }
}

impl FromIterator<Point2> for Extents2 {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Point2>,
    {
        let mut iter = iter.into_iter();
        let Some(init) = iter.next() else { return Self::zero(); };

        let (min, max) = iter.fold((init, init), |(min, max), p| {
            (min.min_all(p), max.max_all(p))
        });

        Self::from_min_max(min, max)
    }
}

/// Build the union of an iterator of [`Extents`].
/// If the iterator is empty, a zero-sized extents about the origin is returned.
impl<P> FromIterator<Self> for Extents<P>
where
    P: Copy + Point + Add,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Self>,
    {
        iter.into_iter()
            .reduce(Extents::union)
            .unwrap_or_else(Extents::zero)
    }
}

impl Envelops<Point3> for Extents3 {
    fn envelops(&self, p: Point3) -> bool {
        let [x, y, z] = p;

        let [mx, my, mz] = self.origin;

        if x < mx || y < my || z < mz {
            return false;
        }

        let [mx, my, mz] = self.max();

        x <= mx && y <= my && z <= mz
    }
}

impl Envelops<Point2> for Extents2 {
    fn envelops(&self, p: Point2) -> bool {
        let [x, y] = p;

        let [mx, my] = self.origin;

        if x < mx || y < my {
            return false;
        }

        let [mx, my] = self.max();

        x <= mx && y <= my
    }
}

pub trait Aabb {
    type Space;

    fn aabb(&self) -> Extents<Self::Space>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;

    type P3 = (f64, f64, f64);
    type E = (P3, P3);

    fn to_p(p: P3) -> Point3 {
        [p.0, p.1, p.2]
    }

    fn to_e((a, b): E) -> Option<Extents3> {
        let a = to_p(a);
        let b = to_p(b);

        a.iter()
            .chain(&b)
            .all(|x| x.is_finite())
            .then(|| Extents3::from_iter([a, b]))
            .filter(|x| x.size.iter().all(|&x| x > 0.0))
    }

    #[quickcheck]
    fn lowering_e3_to_e2(x: E) -> TestResult {
        let Some(e3) = to_e(x) else {
            return TestResult::discard();
        };

        let min = e3.origin.to_p2();
        let max = e3.max().to_p2();

        let e2 = Extents2::from(e3);

        TestResult::from_bool(e2.origin == min && e2.max() == max)
    }

    #[quickcheck]
    fn intersection_consistent(a: E, b: E) -> TestResult {
        let Some(a) = to_e(a) else {
            return TestResult::discard();
        };
        let Some(b) = to_e(b) else {
            return TestResult::discard();
        };
        if a.origin
            .into_iter()
            .zip(b.origin)
            .zip(a.max().into_iter().zip(b.max()))
            .any(|((a, b), (c, d))| a == b || a == c || a == d || b == c || b == d || c == d)
        {
            return TestResult::discard();
        }

        TestResult::from_bool(a.intersects(b) == a.intersection(b).is_some())
    }

    #[quickcheck]
    fn intersects_commutative_e2(a: E, b: E) -> TestResult {
        let Some(a) = to_e(a) else {
            return TestResult::discard();
        };
        let Some(b) = to_e(b) else {
            return TestResult::discard();
        };

        let a = Extents2::from(a);
        let b = Extents2::from(b);

        TestResult::from_bool(a.intersects(b) == b.intersects(a))
    }

    #[quickcheck]
    fn intersects_commutative_e3(a: E, b: E) -> TestResult {
        let Some(a) = to_e(a) else {
            return TestResult::discard();
        };
        let Some(b) = to_e(b) else {
            return TestResult::discard();
        };

        TestResult::from_bool(a.intersects(b) == b.intersects(a))
    }

    #[test]
    fn intersection_tests() {
        let a = Extents3::from_min_max([0.0; 3], [1.0; 3]);
        let b = Extents3 {
            origin: [0., 2., 0.],
            size: Point3::one(),
        };

        assert!(!a.intersects(b));
        assert_eq!(a.intersection(b), None);

        let a = Extents2 {
            origin: [0., -1.0],
            size: Point2::one(),
        };
        let b = Extents2 {
            origin: Point2::zero(),
            size: Point2::one(),
        };

        assert!(a.intersects(b));
        assert!(a.intersection(b).is_some());
    }

    #[test]
    fn cut_smoke_tests() {
        let aabb = Extents3::from_min_max(Point3::zero(), Point3::one());
        let plane = Plane::from([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]);

        let x = aabb.cut(&plane, 1e-7);
        assert_eq!(x.len(), 3);
        assert_eq!(x, &[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5],]);

        let plane = Plane::from([[0.25, 0.0, 0.0], [0.25, 0.5, 0.0], [0.0, 0.0, 0.25]]);

        let x = aabb.cut(&plane, 1e-7);
        assert_eq!(x.len(), 4);
        assert_eq!(
            x,
            &[
                [0.25, 0.0, 0.0],
                [0.25, 1.0, 0.0],
                [0.0, 0.0, 0.25],
                [0.0, 1.0, 0.25],
            ]
        );

        let plane = Plane::from([[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.0, 0.25]]);

        let x = aabb.cut(&plane, 1e-7);
        assert_eq!(x.len(), 0);
    }

    #[test]
    fn expand_test() {
        let e = Extents2::from_min_max(Point2::zero(), Point2::one());

        assert_eq!(
            e.expand(0.5),
            Extents2 {
                origin: [-0.5, -0.5],
                size: [2.0, 2.0]
            }
        );

        assert_eq!(
            e.expand(-0.2),
            Extents2 {
                origin: [0.2, 0.2],
                size: [0.6, 0.6],
            }
        );

        assert_eq!(e.expand(-0.8), Extents2::zero());
        assert_eq!(e.expand(f64::NAN), e);
    }
}
