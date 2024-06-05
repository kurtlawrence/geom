use super::*;

pub struct Polyline3(Vec<Point3>);

impl Polyline3 {
    pub fn new<I>(points: I) -> Result<Self, &'static str>
    where
        I: IntoIterator<Item = Point3>,
    {
        let points = points.into_iter().collect::<Vec<_>>();
        if points.len() < 2 {
            Err("polyline requires 2 or more points to be valid")
        } else {
            Ok(Polyline3(points))
        }
    }
}

impl IntoIterator for Polyline3 {
    type Item = Point3;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}
