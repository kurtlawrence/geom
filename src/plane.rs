use crate::*;

/// Ax + By + Cz = D
#[derive(Debug, Clone)]
pub struct Plane {
    d: f64,
    normal: Point3,
    centroid: Point3,
}

impl Plane {
    pub fn new(centroid: Point3, normal: Point3) -> Self {
        let d = dot_prod(normal, centroid);
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
            let [x, y, z] = p.add(n_centroid); // subtract centroid from point
            xx += x * x; // xx
            xy += x * y; // xy
            xz += x * z; // xz
            yy += y * y; // yy
            yz += y * z; // yz
            zz += z * z; // zz
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
            [detx, xz * yz - xy * zz, xy * yz - xz * yy]
        } else if det_max == dety {
            [xz * yz - xy * zz, dety, xy * xz - yz * xx]
        } else {
            [xy * yz - xz * yy, xy * xz - yz * xx, detz]
        };

        Some(Self::new(centroid, dir))
    }

    pub fn a(&self) -> f64 {
        self.normal[0]
    }

    pub fn b(&self) -> f64 {
        self.normal[1]
    }

    pub fn c(&self) -> f64 {
        self.normal[2]
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
        [self.b(), self.a() * -1.0, 0.0]
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

        let [px, py] = p.to_p2();

        let i = self.d() - self.a() * px - self.b() * py;
        i / self.c()
    }

    /// Returns the **point on the plane** that is the shortest distance from `p`.
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::suspicious_operation_groupings)]
    pub fn shortest_dist(&self, [x, y, z]: Point3) -> Point3 {
        // https://math.stackexchange.com/questions/2758190/shortest-distance-from-point-to-a-plane
        // finding point b such that p + tN = b (point on plane)
        let [a, b, c] = self.normal;
        let d = self.d;

        let t = (d - a * x - b * y - c * z) / (a * a + b * b + c * c);

        [x, y, z].add(self.normal.scale(t)) // p + tN
    }

    /// Returns where **this plane lies** with respect to the `point`.
    ///
    /// **Note the semantics!**. It is answering the sentence 'the plane lies `Above|On|Below` the
    /// point'.
    ///
    /// The notion of 'aboveness' is governed by the plane's normal.
    /// The plane is 'above' the point if the plane-to-point normal runs in the opposite direction,
    /// whilst the plane is 'below' the point if the plane-to-point normal runs in the same
    /// direction.
    ///
    /// The tolerance governs what will be considered coincident within numerical precision.
    ///
    /// # Example
    /// ```rust
    /// # use geom::*;
    /// let plane = Plane::from([
    ///     [0.5, 0.0, 0.0],
    ///     [0.0, 0.5, 0.0],
    ///     [0.0, 0.0, 0.5],
    /// ]);
    ///
    /// // plane lies below (1,1,1)
    /// assert_eq!(plane.lies(Point3::one(), 1e-5), Lies::Below);
    /// // plane lies above (0,0,0)
    /// assert_eq!(plane.lies(Point3::zero(), 1e-5), Lies::Above);
    /// // plane lies on (0.5,0,0)
    /// assert_eq!(plane.lies([0.5, 0.0, 0.0], 1e-5), Lies::On);
    /// ```
    ///
    /// # A note on NaNs
    /// To avoid spurious results, if any structures are poisoned with NaNs (plane or point),
    /// this function _always_ returns `Lies::On`.
    pub fn lies(&self, point: Point3, tolerance: f64) -> Lies {
        // the test is pretty simple, we work out the dot product of the normal and the
        // vector made from the point to the plane.
        // we could use the centroid, however this may run into precision errors if the
        // point is close to the plane but far from the centroid.
        // instead, for a little more work, we work out the vector between the point and
        // the closest point on the plane.
        let cl = self.shortest_dist(point);

        if self
            .normal()
            .into_iter()
            .chain(self.centroid())
            .chain(point)
            .chain(cl)
            .chain(std::iter::once(self.d()))
            .any(|x| x.is_nan())
        {
            return Lies::On;
        }

        let v = point.sub(cl);
        if v.mag() < tolerance {
            Lies::On // point is coincident with plane
        } else if dot_prod(self.normal(), v).is_sign_positive() {
            Lies::Below // plane is below point
        } else {
            Lies::Above
        }
    }
}

impl From<Tri> for Plane {
    fn from(tri: Tri) -> Self {
        let [p0, p1, p2] = tri;
        let n = p0.scale(-1.0);
        let a = p1.add(n); // p1 - p0
        let b = p2.add(n); // p2 - p0
        Plane::new(p0, xprod(a, b))
    }
}

/// Description of relative location between two objects.
///
/// Specifically used for [`Plane::lies`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lies {
    Above,
    On,
    Below,
}

impl Lies {
    /// Invert the semantics.
    ///
    /// # Example
    /// ```rust
    /// # use geom::*;
    /// assert_eq!(Lies::Above.inv(), Lies::Below);
    /// assert_eq!(Lies::On.inv(), Lies::On);
    /// assert_eq!(Lies::Below.inv(), Lies::Above);
    /// ```
    pub fn inv(self) -> Self {
        match self {
            Lies::Above => Lies::Below,
            Lies::On => Lies::On,
            Lies::Below => Lies::Above,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::TestResult;

    #[test]
    fn fit_least_sqs_plane() {
        // going to test a plane which pivots along (-1,1)->(1,-1) at 45° rotation
        let plane = Plane::fit_least_sqs(&[
            [-1.0, -1.0, -(2f64.sqrt())],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 2f64.sqrt()],
        ])
        .unwrap();

        dbg!(&plane);

        assert_eq!(plane.centroid(), [0.0, 0.0, 0.0]);
        let normal = plane.normal;
        let unit_normal = normal
            .unit()
            .add([-0.5, -0.5, 0.5_f64.sqrt()].scale(-1.0))
            .mag();
        assert!(unit_normal.abs() < 1e-11);

        dbg!(normal);
        let x = plane.a() - -11.313708498984761;
        assert!(x.abs() < 1e-11);
        let x = plane.b() - -11.313708498984761;
        assert!(x.abs() < 1e-11);
        let x = plane.c() - 16.0;
        assert!(x.abs() < 1e-11);
        let x = plane.d();
        assert!(x.abs() < 1e-11);

        let strike = plane.strike();
        dbg!(strike);
        let strike = strike
            .unit()
            .add([0.5_f64.sqrt(), -(0.5_f64.sqrt()), 0.0])
            .mag();
        assert!(strike.abs() < 1e-11);

        let dip = plane.dip().unit();
        dbg!(dip);
        let dip = dip.add([0.5, 0.5, 0.5_f64.sqrt()]).mag();
        assert!(dip.abs() < 1e-11);

        let x = plane.register_z([0.5, -0.5]) - 0.0;
        assert!(x.abs() < 1e-11);

        let x = plane.register_z([-0.5, 0.5]) - 0.0;
        assert!(x.abs() < 1e-11);

        let x = plane.register_z([0.5, 0.5]) - 0.5_f64.sqrt();
        assert!(x.abs() < 1e-11);

        let x = plane.register_z([-0.5, -0.5]) - -0.5_f64.sqrt();
        assert!(x.abs() < 1e-11);
    }

    #[quickcheck]
    fn lies_fuzz(ps: ExactFloatsGen<9>) -> TestResult {
        let ps = ps.floats;
        let centroid: Point3 = ps[..3].try_into().unwrap();
        let normal: Point3 = ps[3..6].try_into().unwrap();
        let point: Point3 = ps[6..].try_into().unwrap();

        let plane = Plane::new(centroid, normal);
        let v = point.sub(plane.shortest_dist(point));
        if v.mag().is_infinite() {
            return TestResult::discard(); // can't really test
        }
        let x = match dbg!(plane.lies(point, 1e-7)) {
            Lies::On => v.mag() < 1e-7 || v.mag().is_nan(),
            Lies::Below => {
                v.mag() >= 1e-7
                // normals in same direction
                && (v.unit().add(plane.normal().unit()).mag() - 2.0).abs() < 1e-7
            }
            Lies::Above => {
                v.mag() >= 1e-7
                // normals in opposite direction
                && v.unit().add(plane.normal().unit()).mag() < 1e-7
            }
        };

        TestResult::from_bool(x)
    }
}
