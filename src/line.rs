use crate::*;

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
            .map(|x| [x, 0.0])
    }

    /// The y-intercept, if not vertical.
    pub fn y_intercept(&self) -> Option<Point2> {
        (!self.is_vertical())
            .then(|| self.c / self.b)
            .map(|y| [0.0, y])
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
            let x2 = point[0];
            let diff = x1 - x2;
            if diff.abs() <= f64::EPSILON {
                Equal // co-linear
            } else if diff > 0.0 {
                Greater // x1 is more than x2, x1 is RIGHT of point
            } else {
                Less // x1 is less than x2, x1 is LEFT of point
            }
        } else {
            let [x, y2] = point;
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
        let m = d * [self.a, self.b].mag();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_is_vertical() {
        assert!(Line2 {
            a: 1.0,
            b: 0.0,
            c: 2.0
        }
        .is_vertical());
        assert!(!Line2 {
            a: 1.0,
            b: 0.5,
            c: 2.0
        }
        .is_vertical());
        assert!(!Line2 {
            a: 0.0,
            b: 0.5,
            c: 2.0
        }
        .is_vertical());
    }

    #[test]
    fn cmp_point_to_line() {
        use cmp::Ordering::*;
        // desmos.com/calculator is used

        // +ve gradient
        let line = Line2 {
            a: 1.0,
            b: -1.0,
            c: 1.0,
        };
        assert_eq!(line.pt_cmp([1.0, 0.0]), Equal);
        assert_eq!(line.pt_cmp([2.0, 4.0]), Less);
        assert_eq!(line.pt_cmp([2.0, 0.0]), Greater);

        // -ve gradient
        let line = Line2 {
            a: 3.0,
            b: 2.0,
            c: -4.0,
        };
        assert_eq!(line.pt_cmp([-8.0, 10.0]), Equal);
        assert_eq!(line.pt_cmp([2.0, 4.0]), Less);
        assert_eq!(line.pt_cmp([-5.0, -1.0]), Greater);

        // vertical
        let line = Line2 {
            a: 1.0,
            b: 0.0,
            c: 0.0,
        };
        assert_eq!(line.pt_cmp([0.0, 6.0]), Equal);
        assert_eq!(line.pt_cmp([1.0, 6.0]), Less);
        assert_eq!(line.pt_cmp([-0.5, 6.0]), Greater);

        // horizontal
        let line = Line2 {
            a: 0.0,
            b: 1.0,
            c: 0.0,
        };
        assert_eq!(line.pt_cmp([6.0, 0.0]), Equal);
        assert_eq!(line.pt_cmp([6.0, 1.0]), Less);
        assert_eq!(line.pt_cmp([6.0, -0.5]), Greater);
    }

    #[test]
    fn line2_intercepts() {
        let line = Line2 {
            a: 2.0,
            b: -4.0,
            c: 8.0,
        };
        assert_eq!(line.x_intercept(), Some([4.0, 0.0]));
        assert_eq!(line.y_intercept(), Some([0.0, -2.0]));

        let vert = Line2 {
            a: 1.0,
            b: 0.0,
            c: 4.0,
        };
        assert_eq!(vert.x_intercept(), Some([4.0, 0.0]));
        assert_eq!(vert.y_intercept(), None);

        let horz = Line2 {
            a: 0.0,
            b: 2.0,
            c: 1.0,
        };
        assert_eq!(horz.x_intercept(), None);
        assert_eq!(horz.y_intercept(), Some([0.0, 0.5]));
    }

    #[test]
    fn line2_offset() {
        let line1 = Line2 {
            a: 4.0,
            b: 3.0,
            c: 1.0,
        };
        let line2 = Line2 {
            a: 4.0,
            b: 3.0,
            c: 11.0,
        };
        assert_eq!(line1.offset(2.0), line2);
        let line2 = Line2 {
            a: 4.0,
            b: 3.0,
            c: -9.0,
        };
        assert_eq!(line1.offset(-2.0), line2);

        let line1 = Line2 {
            a: 2.0,
            b: 0.0,
            c: 3.0,
        };
        let line2 = Line2 {
            a: 2.0,
            b: 0.0,
            c: 7.0,
        };
        assert_eq!(line1.offset(2.0), line2);
        let line2 = Line2 {
            a: 2.0,
            b: 0.0,
            c: -1.0,
        };
        assert_eq!(line1.offset(-2.0), line2);

        let line1 = Line2 {
            a: 0.0,
            b: 2.0,
            c: 3.0,
        };
        let line2 = Line2 {
            a: 0.0,
            b: 2.0,
            c: 7.0,
        };
        assert_eq!(line1.offset(2.0), line2);
        let line2 = Line2 {
            a: 0.0,
            b: 2.0,
            c: -1.0,
        };
        assert_eq!(line1.offset(-2.0), line2);
    }
}
