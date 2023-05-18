use std::ops;

pub trait Point: Copy + Sized + IntoIterator<Item = f64> {
    /// Set all the values to this value.
    fn all(v: f64) -> Self;

    /// Set all values to zero.
    fn zero() -> Self {
        Self::all(0.)
    }

    /// Set all values to one.
    fn one() -> Self {
        Self::all(1.)
    }

    /// Scale point by multiplying all dimensions by `scalar`.
    fn scale(self, scalar: f64) -> Self;

    /// Calculate the magnitude of the vector.
    fn mag(self) -> f64 {
        self.into_iter()
            .zip(self)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .sqrt()
    }

    /// Normalise the vector by the magnitude.
    fn unit(self) -> Self {
        self.scale(self.mag().recip())
    }

    /// Return the minimum of each dimension.
    fn min_all(self, b: Self) -> Self {
        xfm(self, b, f64::min)
    }

    /// Return the maximum of each dimension.
    fn max_all(self, b: Self) -> Self {
        xfm(self, b, f64::max)
    }

    /// Return the minimum value of all dimensions.
    fn min_of(self) -> f64 {
        self.into_iter().reduce(f64::min).unwrap()
    }

    /// Return the maximum value of all dimensions.
    fn max_of(self) -> f64 {
        self.into_iter().reduce(f64::max).unwrap()
    }

    /// Perform a transformation on each pair of dimensions.
    fn xfm<F: Fn(f64, f64) -> f64>(self, b: Self, f: F) -> Self;
}

pub trait Add<Rhs = Self> {
    fn add(self, rhs: Rhs) -> Self;
    fn sub(self, rhs: Rhs) -> Self
    where
        Self: Sized + Copy,
        Rhs: Point,
    {
        self.add(rhs.scale(-1.0))
    }
}

/// 2D Point (X,Y).
pub type Point2 = [f64; 2];

/// 3D Point (X,Y,Z).
pub type Point3 = [f64; 3];

impl Add for Point2 {
    fn add(self, rhs: Self) -> Self {
        xfm(self, rhs, ops::Add::add)
    }

    fn sub(self, rhs: Self) -> Self {
        xfm(self, rhs, ops::Sub::sub)
    }
}
impl Point for Point2 {
    fn all(v: f64) -> Self {
        [v; 2]
    }
    fn scale(self, scalar: f64) -> Self {
        self.map(|f| f * scalar)
    }
    fn xfm<F: Fn(f64, f64) -> f64>(self, b: Self, f: F) -> Self {
        let mut x = self.into_iter().zip(b).map(|(a, b)| f(a, b));
        [x.next().unwrap(), x.next().unwrap()]
    }
}

impl Add for Point3 {
    fn add(self, rhs: Self) -> Self {
        Self::xfm(self, rhs, ops::Add::add)
    }

    fn sub(self, rhs: Self) -> Self {
        Self::xfm(self, rhs, ops::Sub::sub)
    }
}
impl Point for Point3 {
    fn all(v: f64) -> Self {
        [v; 3]
    }
    fn scale(self, scalar: f64) -> Self {
        self.map(|f| f * scalar)
    }
    fn xfm<F: Fn(f64, f64) -> f64>(self, b: Self, f: F) -> Self {
        let mut x = self.into_iter().zip(b).map(|(a, b)| f(a, b));
        [x.next().unwrap(), x.next().unwrap(), x.next().unwrap()]
    }
}

impl Add<Point2> for Point3 {
    fn add(self, rhs: Point2) -> Self {
        let rhs: Point3 = rhs.with_z(0.);
        self.add(rhs)
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
impl ToPoint2 for &Point2 {
    fn to_p2(self) -> Point2 {
        *self
    }
}
impl ToPoint2 for Point3 {
    fn to_p2(self) -> Point2 {
        let [x, y, _] = self;
        [x, y]
    }
}
impl ToPoint2 for &Point3 {
    fn to_p2(self) -> Point2 {
        (*self).to_p2()
    }
}

pub trait WithZ: ToPoint2 + Sized {
    fn with_z(self, z: f64) -> Point3 {
        let [x, y] = self.to_p2();
        [x, y, z]
    }
}

impl WithZ for Point2 {}
impl WithZ for Point3 {}

pub trait WithX {
    fn with_x(self, x: f64) -> Self;
}

impl WithX for Point2 {
    fn with_x(mut self, x: f64) -> Self {
        self[0] = x;
        self
    }
}
impl WithX for Point3 {
    fn with_x(mut self, x: f64) -> Self {
        self[0] = x;
        self
    }
}

pub trait WithY {
    fn with_y(self, y: f64) -> Self;
}

impl WithY for Point2 {
    fn with_y(mut self, y: f64) -> Self {
        self[1] = y;
        self
    }
}
impl WithY for Point3 {
    fn with_y(mut self, y: f64) -> Self {
        self[1] = y;
        self
    }
}

pub fn dot_prod(a: Point3, b: Point3) -> f64 {
    a.into_iter().zip(b).map(|(a, b)| a * b).sum()
}

#[allow(clippy::many_single_char_names)]
pub fn xprod(a: Point3, b: Point3) -> Point3 {
    let [ax, ay, az] = a;
    let [bx, by, bz] = b;
    let x = ay * bz - az * by;
    let y = az * bx - ax * bz;
    let z = ax * by - ay * bx;
    [x, y, z]
}

pub fn grade(p3: Point3) -> f64 {
    p3[2] / len_xy(p3)
}

pub fn len_xy(p: impl ToPoint2) -> f64 {
    p.to_p2().mag()
}

pub fn zero_len_xy(p: impl ToPoint2) -> bool {
    let [x, y] = p.to_p2();
    x.abs() < 1e-7 && y.abs() < 1e-7
}

/// Apply an ordering to points by testing each x,y,z.
pub fn ordpt<P: Point>(a: P, b: P) -> std::cmp::Ordering {
    use std::cmp::Ordering::Equal;

    a.into_iter().zip(b).fold(
        Equal,
        |o, (a, b)| {
            if o == Equal {
                a.total_cmp(&b)
            } else {
                o
            }
        },
    )
}

/// Helper function which effectively transforms to [`Point::xfm`].
#[inline(always)]
pub fn xfm<P: Point, F: Fn(f64, f64) -> f64>(a: P, b: P, f: F) -> P {
    P::xfm(a, b, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_adding() {
        let p = [0.0, 1.0].add([3.0, 1.0]);
        assert_eq!(p, [3.0, 2.0]);

        let p = [0.0, 1.0, 5.0].add([3.0, 1.0]);
        assert_eq!(p, [3.0, 2.0, 5.0]);

        let p = [0.0, 1.0, 5.0].add([3.0, 1.0, 5.0]);
        assert_eq!(p, [3.0, 2.0, 10.0]);
    }

    #[test]
    fn point_scaling() {
        let p = [0.0, 1.0].scale(2.0);
        assert_eq!(p, [0.0, 2.0]);

        let p = [-2.0, 0.5, 3.0].scale(-0.5);
        assert_eq!(p, [1.0, -0.25, -1.5]);
    }

    #[test]
    fn to_point_testing() {
        assert_eq!([0.0, 1.0].to_p2(), [0.0, 1.0]);
        assert_eq!([0.0, 1.0, 2.0].to_p2(), [0.0, 1.0]);
    }

    #[test]
    fn xproduct_test() {
        let v = xprod([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(v, [0.0, 0.0, 1.0]);

        let v = xprod([1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]);
        assert_eq!(v, [0.0, -0.0, 2.0]);
    }

    #[test]
    fn grade_testing() {
        let x = grade([0.0, 10.0, 1.0]) - 0.1;
        assert!(x.abs() < 1e-11);

        let x = grade([0.0, 10.0, -1.0]) - -0.1;
        assert!(x.abs() < 1e-11);

        let x = grade([3.0, 4.0, 1.0]) - 0.2;
        assert!(x.abs() < 1e-11);

        let x = grade([3.0, 4.0, -1.0]) - -0.2;
        assert!(x.abs() < 1e-11);
    }

    #[test]
    fn len_xy_testing() {
        let x = len_xy([3.0, 4.0, 1.0]) - 5.0;
        assert!(x.abs() < 1e-11);

        assert_eq!(zero_len_xy([0.0, 0.0, 1.0]), true);
        assert_eq!(zero_len_xy([0.0, 1.0, 1.0]), false);
        assert_eq!(zero_len_xy([1.0, 0.0, 1.0]), false);
        assert_eq!(zero_len_xy([1.0, 1.0, 1.0]), false);
    }

    #[test]
    fn mag_testing() {
        let m = [3.0, 4.0].mag() - 5.0;
        assert!(m.abs() < 1e-11);

        let m = [3.0, -4.0].mag() - 5.0;
        assert!(m.abs() < 1e-11);

        let m = [-3.0, 4.0].mag() - 5.0;
        assert!(m.abs() < 1e-11);

        let m = [2.0, 3.0, 6.0].mag() - 7.0;
        assert!(m.abs() < 1e-11);

        let m = [2.0, -3.0, 6.0].mag() - 7.0;
        assert!(m.abs() < 1e-11);

        let m = [-2.0, -3.0, 6.0].mag() - 7.0;
        assert!(m.abs() < 1e-11);

        let m = [-2.0, -3.0, -6.0].mag() - 7.0;
        assert!(m.abs() < 1e-11);
    }

    #[test]
    fn unit_vector() {
        let u = [2.0, 0.0].unit();
        assert_eq!(u, [1.0, 0.0]);

        let u = [0.0, 0.0, 2.0].unit();
        assert_eq!(u, [0.0, 0.0, 1.0]);
    }
}
