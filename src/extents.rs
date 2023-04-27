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
            || self
                .origin
                .into_iter()
                .zip(other.max())
                .any(|(o, c)| o >= c);

        !outside
    }
}

impl Extents3 {
    pub fn from_min_max(min: Point3, max: Point3) -> Self {
        let size = max.sub(min);

        Self { origin: min, size }
    }

    /// A three-dimensional AABB has volume!
    pub fn volume(&self) -> f64 {
        let [w, d, h] = self.size;
        w * d * h
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
        todo!()
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
}
