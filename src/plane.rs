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
}

impl From<&Tri> for Plane {
    fn from(tri: &Tri) -> Self {
        let n = tri.0.scale(-1.0);
        let a = tri.1.add(n); // p1 - p0
        let b = tri.2.add(n); // p2 - p0
        Plane::new(tri.0, xprod(a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
