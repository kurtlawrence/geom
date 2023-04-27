use crate::*;

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

    pub fn iter(&self) -> impl ExactSizeIterator<Item = Point2> + '_ {
        self.0.iter().copied()
    }
}

/// Test if point is _inside_ the polygon.
impl<O: ToPoint2> Envelops<O> for Polygon2 {
    fn envelops(&self, p: O) -> bool {
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
