use crate::*;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Polygon2(Vec<Point2>);

impl Polygon2 {
    pub fn new<I, P>(points: I) -> Result<Self, &'static str>
    where
        I: IntoIterator<Item = P>,
        P: ToPoint2,
    {
        let points = points.into_iter().map(ToPoint2::to_p2).collect::<Vec<_>>();
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

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Point2> + '_ {
        self.0.iter().copied()
    }
}

impl Area for Polygon2 {
    /// 2D plan area.
    ///
    /// # Example
    /// ```rust
    /// use geom::*;
    /// let p = Polygon2::new([
    ///     [0.0, 0.0],
    ///     [2.0, 0.0],
    ///     [2.0, 3.0],
    ///     [0.0, 3.0]
    /// ]).unwrap();
    ///
    /// assert!((p.area() - 6.0).abs() < 1e-3);
    /// ```
    fn area(&self) -> f64 {
        // https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
        self.pts()
            .iter()
            .zip(self.pts().iter().skip(1))
            .chain(std::iter::once((
                self.pts().last().unwrap(),
                &self.pts()[0],
            )))
            .map(|([ax, ay], [bx, by])| ax * by - ay * bx)
            .sum::<f64>()
            .abs()
            * 0.5
    }
}

/// Test if a point is _inside_ a polygon, represented by a vector of points.
///
/// Uses a winding counting number test.
///
/// # Panics
/// Panics if the number of points is < 3.
pub fn point_inside<P, O>(ps: &[P], point: O) -> bool
where
    O: ToPoint2,
    for<'a> &'a P: ToPoint2,
{
    if ps.len() < 3 {
        panic!("requires at least 3 points to form a polygon");
    }

    use std::iter::*;
    // uses the counting winding number test!

    /// Test if a point is on a line.
    /// > 0 : p is left of line
    /// = 0 : p is on line
    /// < 0 : p is right of line
    fn on((from, to): (Point2, Point2), p: Point2) -> f64 {
        let f = from.scale(-1.0);
        let [ax, ay] = to.add(f);
        let [bx, by] = p.add(f);
        ax * by - bx * ay
    }

    let point = point.to_p2();
    let one_off = ps.len() - 1; // shouldn't underflow since we maintain > 3 pts invariant
    let froms = &ps[..one_off];
    let tos = &ps[1..];

    let segments = froms
        .iter()
        .zip(tos.iter())
        .chain(once((&ps[one_off], &ps[0])))
        .map(|(a, b)| (a.to_p2(), b.to_p2()));

    let mut winding = 0;

    let pointy = point[1];
    for (from, to) in segments {
        let fromy = from[1];
        let toy = to[1];
        if fromy <= pointy {
            if toy > pointy && on((from, to), point) > 0.0 {
                winding += 1;
            }
        } else if toy <= pointy && on((from, to), point) < 0.0 {
            winding -= 1;
        }
    }

    winding != 0
}

/// Test if point is _inside_ the polygon.
impl<O: ToPoint2> Envelops<O> for Polygon2 {
    fn envelops(&self, p: O) -> bool {
        point_inside(&self.0, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_inside_testing() {
        let polygon = Polygon2::new(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
                .iter()
                .copied(),
        )
        .unwrap();
        assert_eq!(polygon.envelops([0.5, 0.5]), true);
        assert_eq!(polygon.envelops([0.9, 0.9]), true);
        assert_eq!(polygon.envelops([0.9999999, 0.999999]), true);
        assert_eq!(polygon.envelops([0.00001, 0.00001]), true);
        assert_eq!(polygon.envelops([0.3, 0.1]), true);

        assert_eq!(polygon.envelops([-0.3, 0.1]), false);
        assert_eq!(polygon.envelops([0.3, -0.1]), false);
        assert_eq!(polygon.envelops([1.1, -0.1]), false);
        assert_eq!(polygon.envelops([1.1, 0.1]), false);
        assert_eq!(polygon.envelops([0.5, 1.1]), false);
        assert_eq!(polygon.envelops([0.5, -1.1]), false);
        assert_eq!(polygon.envelops([1.5, 1.1]), false);

        // notice that when on upper boundary it is false
        assert_eq!(polygon.envelops([0.0, 0.0]), true);
        assert_eq!(polygon.envelops([0.5, 0.0]), true);
        assert_eq!(polygon.envelops([1.0, 0.0]), false);
        assert_eq!(polygon.envelops([1.0, 0.5]), false);
        assert_eq!(polygon.envelops([1.0, 1.0]), false);
        assert_eq!(polygon.envelops([0.5, 1.0]), false);
        assert_eq!(polygon.envelops([0.0, 1.0]), false);
        assert_eq!(polygon.envelops([0.0, 0.5]), true);
    }
}
