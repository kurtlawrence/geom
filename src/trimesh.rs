use crate::*;

/// Triangle represented by 3 points (A, B, C).
pub type Tri = [Point3; 3];

impl Aabb for Tri {
    type Space = Point3;
    fn aabb(&self) -> Extents<Self::Space> {
        Extents::from_iter(*self)
    }
}

/// A triangle mesh.
///
/// `PartialEq` is _derived_ but does _exact_ equality including structural equality. This is
/// **not** the same as value equality (it is a _subset_ of it) so `PartialEq` should not be used
/// for value equality.
#[derive(Debug, PartialEq, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TriMesh {
    /// The _distinct_ points.
    points: Vec<Point3>,
    /// Each _triangle_ is a triplet of points.
    ///
    /// Each tuple entry is the _index_ back into the `points`.
    /// This is done to save space so a 'triangle' takes up 12 bytes,
    /// and points are not duplicated. (A point itself takes **24 bytes**).
    triangles: Vec<(u32, u32, u32)>,
}

impl TriMesh {
    pub fn from_raw(points: Vec<Point3>, triangles: Vec<(u32, u32, u32)>) -> Self {
        let mut x = Self { points, triangles };
        x.consolidate();
        x
    }

    pub fn decompose(self) -> (Vec<Point3>, Vec<(u32, u32, u32)>) {
        let Self { points, triangles } = self;
        (points, triangles)
    }

    pub fn point_len(&self) -> usize {
        self.points.len()
    }

    pub fn tri_len(&self) -> usize {
        self.triangles.len()
    }

    pub fn points(&self) -> &[Point3] {
        &self.points
    }

    pub fn tri_indices(&self) -> &[(u32, u32, u32)] {
        &self.triangles
    }

    pub fn tris(&self) -> impl ExactSizeIterator<Item = Tri> + '_ {
        self.triangles.iter().map(move |&(a, b, c)| {
            [
                self.points[a as usize],
                self.points[b as usize],
                self.points[c as usize],
            ]
        })
    }

    pub fn extend(&mut self, tris: impl Iterator<Item = Tri>) {
        // call into the extend_trimesh module since there are perf improvements there
        extend_trimesh::extend(self, tris)
    }

    /// Returns a set of polygons that trace the outside of the triangle mesh.
    /// If the mesh is closed this would be an empty vector.
    pub fn outlines(&self) -> Vec<Vec<Point3>> {
        // the outlines algorithm is as follows:
        // 1. get set of _edges_ that only have _one_ tri,
        // 2. join edges together
        // the main algorithm is specified in the module `outline` to enforce some checking and
        // performance code

        // 1. get free edges
        let free_edges = outline::free_edges(self);

        // 2. join edges together
        let outlines = outline::get_outlines(free_edges.into_iter().collect());

        outlines
            .into_iter()
            .map(|o| {
                let take = if o.front() == o.back() {
                    o.len().saturating_sub(1)
                } else {
                    o.len()
                };

                o.into_iter()
                    .take(take)
                    .map(|idx| self.points[idx as usize])
                    .collect()
            })
            .collect()
    }

    /// Ensures that there are **not** additional points that do not have faces by removing these
    /// extra points.
    pub fn consolidate(&mut self) {
        let len = self.point_len();
        let mut used = vec![false; len];
        for (a, b, c) in self.triangles.iter().copied() {
            used[a as usize] = true;
            used[b as usize] = true;
            used[c as usize] = true;
        }

        for idx in (0..len).rev().filter(|&x| !used[x]) {
            // remove at idx (reversed so order is maintained)
            // have to _decrement_ all idx references _above_ idx =/
            self.points.remove(idx);
            let idx = idx as u32;
            for x in self.triangles.iter_mut() {
                if x.0 > idx {
                    x.0 -= 1;
                }
                if x.1 > idx {
                    x.1 -= 1;
                }
                if x.2 > idx {
                    x.2 -= 1;
                }
            }
        }
    }
}

impl From<&Grid> for TriMesh {
    fn from(grid: &Grid) -> Self {
        // since we are building from a grid there are some optimisations we can make
        // 1. Grid::points can be collected as Vec<Point3>, at the same time, a map can be made of
        //    (x,y): index
        // 2. Centre points can be added progressively, since they are known to be distinct

        // build points store + lookup
        let len = grid.len();
        let seed = (
            Vec::with_capacity(len * 2), // account for centre points
            HashMap::with_capacity_and_hasher(len, Default::default()),
        );
        let (mut points, pmap) = grid.points().fold(seed, |(mut pts, mut map), p| {
            let idx = pts.len();
            pts.push(p.p3());
            map.insert(p.loc(), idx);
            (pts, map)
        });

        fn centre(a: impl Into<Point3>, b: impl Into<Point3>) -> Point3 {
            a.into().add(b.into()).scale(0.5)
        }

        let i = |p: GridPoint<f64>| *pmap.get(&p.loc()).expect("needs to be in map") as u32;

        // work through each grid cell -- serial
        let mut triangles = Vec::with_capacity(grid.len() * 4);
        let locs = (0..grid.y_count().saturating_sub(1))
            .flat_map(|y| (0..grid.x_count().saturating_sub(1)).map(move |x| (x, y)));
        for (x, y) in locs {
            let ps = (
                grid.get(x, y),
                grid.get(x + 1, y),
                grid.get(x + 1, y + 1),
                grid.get(x, y + 1),
            );

            match ps {
                (Some(p1), Some(p2), Some(p3), Some(p4)) => {
                    // all four points available!
                    let c = p1.p3().add(p2.p3()).add(p3.p3()).add(p4.p3()).scale(0.25);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p1), ci, i(p2))); // bottom
                    triangles.push((i(p2), ci, i(p3))); // right
                    triangles.push((i(p3), ci, i(p4))); // top
                    triangles.push((i(p4), ci, i(p1))); // left
                }
                (None, Some(p2), Some(p3), Some(p4)) => {
                    let c = centre(p2, p4);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p2), ci, i(p3))); // right
                    triangles.push((i(p3), ci, i(p4))); // top
                }
                (Some(p1), None, Some(p3), Some(p4)) => {
                    let c = centre(p1, p3);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p3), ci, i(p4))); // top
                    triangles.push((i(p4), ci, i(p1))); // left
                }
                (Some(p1), Some(p2), None, Some(p4)) => {
                    let c = centre(p2, p4);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p1), ci, i(p2))); // bottom
                    triangles.push((i(p4), ci, i(p1))); // left
                }
                (Some(p1), Some(p2), Some(p3), None) => {
                    let c = centre(p1, p3);
                    let ci = points.len() as u32;
                    points.push(c);

                    triangles.push((i(p1), ci, i(p2))); // bottom
                    triangles.push((i(p2), ci, i(p3))); // right
                }
                _ => (),
            }
        }

        triangles.shrink_to_fit();
        points.shrink_to_fit();
        Self { points, triangles }
    }
}

impl FromIterator<Tri> for TriMesh {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Tri>,
    {
        let mut t = TriMesh {
            points: Vec::new(),
            triangles: Vec::new(),
        };
        t.extend(iter.into_iter());
        t
    }
}

/// Test if a point is **inside** a solid [`TriMesh`].
///
/// If you are wanting to test that a point lays _on_ a surface, use envelops with a [`Point2`].
impl Envelops<Point3> for TriMesh {
    fn envelops(&self, p: Point3) -> bool {
        todo!()
    }
}

/// Test if a point lays **on** a [`TriMesh`].
///
/// Note this solely tests for 2D intersection.
impl Envelops<Point2> for TriMesh {
    fn envelops(&self, p: Point2) -> bool {
        if Extents2::from(self.aabb()).envelops(p) {
            self.tris().any(|tri| polygon::point_inside(&tri, p))
        } else {
            false
        }
    }
}

impl Aabb for TriMesh {
    type Space = Point3;
    fn aabb(&self) -> Extents3 {
        self.points().iter().copied().collect()
    }
}

/// Same point, with tolerance.
fn same_point(a: Point3, b: Point3) -> bool {
    const T: f64 = 1e-7;
    a.xfm(b, |a, b| (a - b).abs()).into_iter().all(|f| f < T)
}

mod outline {
    use super::*;
    use std::{collections::VecDeque, hash::*};

    #[derive(PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord)]
    pub struct Edge(pub u32, pub u32);

    impl Edge {
        #[inline(always)]
        fn ord(self) -> Self {
            if self.0 <= self.1 {
                self
            } else {
                Edge(self.1, self.0)
            }
        }

        #[inline(always)]
        fn as_u64(&self) -> u64 {
            ((self.0 as u64) << 32) | self.1 as u64
        }
    }
    impl From<(u32, u32)> for Edge {
        #[inline(always)]
        fn from((a, b): (u32, u32)) -> Self {
            Edge(a, b)
        }
    }
    #[allow(clippy::derive_hash_xor_eq)]
    impl Hash for Edge {
        #[inline(always)]
        fn hash<H: Hasher>(&self, hasher: &mut H) {
            hasher.write_u64(self.as_u64())
        }
    }

    pub fn free_edges(mesh: &TriMesh) -> HashSet<Edge> {
        let tris = mesh.triangles.iter().copied();

        let mut set: HashSet<Edge> =
            HashSet::with_capacity_and_hasher(mesh.tri_len(), Default::default());
        let mut ins = |e: Edge| {
            if !set.remove(&e) {
                set.insert(e);
            }
        };

        for (a, b, c) in tris {
            ins(Edge(a, b).ord());
            ins(Edge(a, c).ord());
            ins(Edge(b, c).ord());
        }

        set
    }

    fn join_edges(edges: &mut Vec<Edge>) -> VecDeque<u32> {
        // worst case is O(n²)
        let mut v = VecDeque::with_capacity(edges.len());
        match edges.pop() {
            Some(Edge(a, b)) => {
                v.push_back(a);
                v.push_back(b);
            }
            None => return v,
        }

        // the get algorithm uses swap_remove to linearly search through edges
        let mut get = |pos| {
            let mut r = None;
            // search backwards, since swap_remove swaps in last element
            for i in (0..edges.len()).rev() {
                let Edge(a, b) = edges.swap_remove(i); // last goes to i'th
                if a == pos {
                    r = Some(b); // a matches pos, so b is next point
                } else if b == pos {
                    r = Some(a); // b matches pos, so a is next point
                } else {
                    edges.push(Edge(a, b)); // neither matches, push edge back
                }
                if r.is_some() {
                    break; // found it so break
                }
            }

            r
        };

        let (mut has_front, mut has_back) = (true, true);

        loop {
            if !has_front && !has_back {
                break;
            }

            if has_front {
                let front = *v.front().expect("will have front");
                match get(front) {
                    Some(x) => v.push_front(x),
                    None => has_front = false,
                }
            }

            if has_back {
                let back = *v.back().expect("will have back");
                match get(back) {
                    Some(x) => v.push_back(x),
                    None => has_back = false,
                }
            }
        }

        v
    }

    pub fn get_outlines(free_edges: Vec<Edge>) -> Vec<VecDeque<u32>> {
        let mut edges = free_edges;
        let mut v = Vec::new();

        while !edges.is_empty() {
            let shape = join_edges(&mut edges);
            v.push(shape);
        }

        v
    }
}

mod extend_trimesh {
    use super::*;
    use std::cmp::Ordering;

    // define point wrapper which handles the ordering + equality with tolerance
    struct Pt {
        p: Point3,
        i: u32,
    }
    impl PartialEq for Pt {
        fn eq(&self, rhs: &Self) -> bool {
            same_point(self.p, rhs.p)
        }
    }
    impl Eq for Pt {}
    impl PartialOrd for Pt {
        fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
            Some(self.cmp(rhs))
        }
    }
    impl Ord for Pt {
        #[allow(clippy::float_cmp)]
        fn cmp(&self, rhs: &Self) -> Ordering {
            if self.eq(rhs) {
                // expensive to check first, but handles the case for equality with tolerance
                return Ordering::Equal;
            }

            ordpt(self.p, rhs.p)
        }
    }

    pub fn extend(mesh: &mut TriMesh, tris: impl Iterator<Item = Tri>) {
        // first collect the current points->idx into a BTreeSet
        let mut ordset = std::collections::BTreeSet::new();
        ordset.extend(
            mesh.points
                .iter()
                .enumerate()
                .map(|(i, &p)| Pt { p, i: i as u32 }),
        );

        let TriMesh { points, triangles } = mesh;

        let mut get_or_add = |p: Point3| {
            let pt = Pt {
                p,
                i: points.len() as u32,
            };
            match ordset.get(&pt) {
                Some(idx) => idx.i,
                None => {
                    let i = pt.i;
                    points.push(pt.p);
                    ordset.insert(pt);
                    i
                }
            }
        };

        for [a, b, c] in tris {
            let t = (get_or_add(a), get_or_add(b), get_or_add(c));
            triangles.push(t);
        }
    }
}

mod contours {
    use super::*;

    impl TriMesh {
        pub fn contour(&self, spacing: f64) -> Vec<Polyline> {
            use rayon::prelude::*;

            let aabb = self.aabb();
            let (min, max) = (aabb.origin, aabb.max());
            let start = (min[2] / spacing).floor() * spacing;
            let end = (max[2] / spacing).floor() * spacing;
            let steps = ((end - start) / spacing).trunc() as usize;

            let faces = self.faces();

            (0..=steps)
                .into_par_iter()
                .map(|i| {
                    let rl = start + i as f64 * spacing;
                    contour_rl(&faces, rl)
                })
                .reduce(Vec::new, |mut a, b| {
                    a.extend(b.into_iter());
                    a
                })
        }

        pub fn contour_rl(&self, rl: f64) -> Vec<Polyline> {
            let faces = self.faces();
            contour_rl(&faces, rl)
        }

        fn faces(&self) -> Vec<Face> {
            let mut faces = self.tris().map(Face::from).collect::<Vec<_>>();
            faces.sort_unstable_by(|a, b| a.minz.partial_cmp(&b.minz).expect("point isn't a NaN"));
            faces
        }
    }

    type Polyline = Vec<Point3>;
    type Edge3D = (Point3, Point3);
    type Edge = (usize, usize);

    struct Face {
        minz: f64,
        maxz: f64,
        tri: Tri,
    }

    impl From<Tri> for Face {
        fn from(tri: Tri) -> Self {
            let [[_, _, a], [_, _, b], [_, _, c]] = tri;
            let minz = a.min(b).min(c);
            let maxz = a.max(b).max(c);
            Self { minz, maxz, tri }
        }
    }

    /// **Assumes `faces` is sorted by minimum z!** Required for search space reduction.
    fn contour_rl(faces: &[Face], rl: f64) -> Vec<Polyline> {
        let gt = faces
            .binary_search_by(|f| f.minz.partial_cmp(&rl).expect("not NaN"))
            .unwrap_or_else(|e| e);
        if gt >= faces.len() {
            return Vec::new();
        }

        // reduce search space (everything > gt does not intersect rl)
        let tris = faces[..=gt].iter().filter(|f| rl <= f.maxz).map(|x| x.tri);
        let edges = contour_edges(tris, rl);
        let (mut edges, points) = index_edges(edges);
        sort_and_dedup_edges(&mut edges);
        let splits = join_edges(&mut edges);

        let mut start = 0;
        splits
            .into_iter()
            .map(|s| {
                let polyline = flatten_edges(&edges[start..s])
                    .map(|i| points[i])
                    .collect::<Polyline>();
                start = s;
                polyline
            })
            .collect()
    }

    fn contour_edges<I>(tris: I, rl: f64) -> Vec<Edge3D>
    where
        I: Iterator<Item = Tri>,
    {
        tris.filter_map(|t| contour_tri(t, rl)).collect()
    }

    fn contour_tri(tri: Tri, rl: f64) -> Option<Edge3D> {
        let [s1, s2, d] = split_tri(tri, rl)?;
        let a = point_at_rl(d, s1, rl);
        let b = point_at_rl(d, s2, rl);
        Some((a, b))
    }

    /// Splits the tri into 2 points on one side of the rl, and one point on the other side. The
    /// order/value does not matter, only that two points must be on same side.
    /// Returns `None` if a split does not work (all points on one side), or if all points are on
    /// the same rl.
    /// Returns `Some([same, same, different])`.
    fn split_tri(tri: Tri, rl: f64) -> Option<Tri> {
        const T: f64 = 1e-5;
        let mut arr = tri;
        if arr.iter().all(|[_, _, z]| (z - rl).abs() < T) {
            return None; // all == rl
        }

        let (a, b, c) = {
            let mut x = arr
                .iter()
                .map(|[_, _, z]| ((z - rl).signum() - 1.0).abs() < 1e-7);
            (x.next(), x.next(), x.next())
        };

        if a == b && b == c {
            None // all on same side of rl
        } else {
            if a == c {
                arr.swap(1, 2); // move 2nd to 3rd
            } else if b == c {
                arr.swap(0, 2); // swap 1st and last
            }

            #[allow(clippy::float_cmp)]
            {
                debug_assert_eq!((arr[0][2] - rl).signum(), (arr[1][2] - rl).signum());
                debug_assert_ne!((arr[0][2] - rl).signum(), (arr[2][2] - rl).signum());
                debug_assert_ne!((arr[1][2] - rl).signum(), (arr[2][2] - rl).signum());
            }
            Some(arr)
        }
    }

    fn point_at_rl(p1: Point3, p2: Point3, rl: f64) -> Point3 {
        let v = p2.sub(p1);
        let r = (rl - p1[2]) / v[2];
        p1.add(v.scale(r))
    }

    /// Each `Edge` has been _ordered_, but the `Vec`tor has **not**.
    fn index_edges(edges: Vec<Edge3D>) -> (Vec<Edge>, Vec<Point3>) {
        let mut pts = Vec::new();
        let mut es = Vec::with_capacity(edges.len());
        for (a, b) in edges {
            let a = get_or_add(&mut pts, a);
            let b = get_or_add(&mut pts, b);

            if a == b {
                continue;
            }

            // order the edges here
            let e = if a < b { (a, b) } else { (b, a) };
            es.push(e);
        }

        (es, pts)
    }

    fn get_or_add(pts: &mut Vec<Point3>, p: Point3) -> usize {
        match pts
            .iter()
            .enumerate()
            .find(|x| same_point(p, *x.1))
            .map(|x| x.0)
        {
            Some(idx) => idx,
            None => {
                let idx = pts.len();
                pts.push(p);
                idx
            }
        }
    }

    fn sort_and_dedup_edges(edges: &mut Vec<Edge>) {
        edges.sort_unstable();
        edges.dedup();
    }

    /// Orders `edges` such that `e0.1 == e1.0`
    /// If edges do not join, this index defines the 'split' which dictates a _new polyline_.
    /// Edges are 'flipped' if matched on opposite orientation.
    fn join_edges(edges: &mut [Edge]) -> Vec<usize> {
        let mut is = Vec::new();
        let mut slice = edges;
        let mut offset = 0;

        while !slice.is_empty() {
            let i = join_edges_slice(slice);
            offset += i;
            is.push(offset);
            slice = &mut slice[i..];
        }

        is
    }

    fn join_edges_slice(edges: &mut [Edge]) -> usize {
        fn rev(edges: &mut [Edge]) {
            edges.reverse();
            for e in edges {
                *e = (e.1, e.0); // flip the edges around
            }
        }

        if edges.is_empty() {
            panic!("called with empty slice");
        }

        let len = edges.len();
        for i in 0..len {
            let mut front = edges[0].0;
            let mut back = edges[i].1;
            let mut found = false;
            let swp = i + 1;

            for j in swp..len {
                let (a, b) = edges[j];

                if a == front || b == front {
                    // this edge matches the _front_. We need to rev and flip front/back
                    rev(&mut edges[..swp]); // reverses up to (not incl) j
                    std::mem::swap(&mut front, &mut back);
                    // double check that this rev/swap does what I expect it to
                    debug_assert_eq!(front, edges[0].0);
                    debug_assert_eq!(back, edges[i].1);
                }

                if a == back {
                    // back.1 == e.0, which is the best case!
                    // break the inner loop and let it progress
                    edges.swap(swp, j);
                    found = true;
                    break;
                } else if b == back {
                    // back.1 == e.1, need to flip e around
                    // break the inner loop and let it progress
                    edges[j] = (b, a);
                    edges.swap(swp, j);
                    found = true;
                    break;
                }
            }

            if !found {
                return swp;
            }
            if edges[swp].1 == front {
                return swp + 1;
            }
        }

        len
    }

    fn flatten_edges(edges: &[Edge]) -> impl Iterator<Item = usize> + '_ {
        edges
            .get(0)
            .map(|x| x.0)
            .into_iter()
            .chain(edges.iter().map(|x| x.1))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn join_edges_slice_2() {
            let mut edges = [(0, 1), (1, 2)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(0, 1), (1, 2)]);

            let mut edges = [(0, 1), (2, 0)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(1, 0), (0, 2)]);

            let mut edges = [(0, 1), (2, 1)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(0, 1), (1, 2)]);
        }

        #[test]
        fn join_edges_slice_3() {
            let mut edges = [(0, 1), (1, 2), (2, 3)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 3);
            assert_eq!(edges, [(0, 1), (1, 2), (2, 3)]);

            let mut edges = [(0, 1), (2, 0), (2, 3)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 3);
            assert_eq!(edges, [(1, 0), (0, 2), (2, 3)]);

            let mut edges = [(0, 1), (3, 4), (2, 1)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(0, 1), (1, 2), (3, 4)]);
        }

        #[test]
        fn join_edges_slice_4() {
            let mut edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 4);
            assert_eq!(edges, [(3, 2), (2, 1), (1, 0), (0, 3)]);

            let mut edges = [(0, 1), (2, 0), (2, 3), (1, 4)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 4);
            assert_eq!(edges, [(3, 2), (2, 0), (0, 1), (1, 4)]);

            let mut edges = [(0, 1), (3, 4), (2, 1), (5, 6)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 2);
            assert_eq!(edges, [(0, 1), (1, 2), (3, 4), (5, 6)]);

            let mut edges = [(0, 1), (3, 4), (0, 2), (1, 2)];
            let i = join_edges_slice(&mut edges);
            assert_eq!(i, 3);
            assert_eq!(edges, [(2, 0), (0, 1), (1, 2), (3, 4)]);
        }

        #[test]
        fn join_edges_01() {
            let mut edges = [(0, 1), (2, 1)];
            let v = join_edges(&mut edges);
            assert_eq!(v, vec![2]);
            assert_eq!(edges, [(0, 1), (1, 2)]);
        }

        #[test]
        fn join_edges_02() {
            let mut edges = [(0, 1), (2, 1), (3, 4)];
            let v = join_edges(&mut edges);
            assert_eq!(v, vec![2, 3]);
            assert_eq!(edges, [(0, 1), (1, 2), (3, 4)]);
        }

        #[test]
        fn join_edges_03() {
            let mut edges = [(0, 1), (3, 4), (2, 1), (5, 6)];
            let v = join_edges(&mut edges);
            assert_eq!(v, vec![2, 3, 4]);
            assert_eq!(edges, [(0, 1), (1, 2), (3, 4), (5, 6)]);

            let mut edges = [(0, 1), (3, 4), (0, 2), (1, 2)];
            let v = join_edges(&mut edges);
            assert_eq!(v, vec![3, 4]);
            assert_eq!(edges, [(2, 0), (0, 1), (1, 2), (3, 4)]);
        }

        fn dummy_tri() -> TriMesh {
            let mut g = Grid::new([1.0, 2.0], 4, 3, 15.0);
            for (z, (x, y)) in (0..4).flat_map(|x| (0..3).map(move |y| (x, y))).enumerate() {
                g.set(x, y, z as f64);
            }

            TriMesh::from(&g)
        }

        #[test]
        fn contour_rl_test() {
            let t = &dummy_tri();
            let c = t.contour_rl(-1.0);
            assert!(c.is_empty());

            let c = t.contour_rl(5.0);
            assert_eq!(
                &c,
                &[vec![
                    [16.0, 32.0, 5.0],
                    [19.75, 20.75, 5.0],
                    [21.0, 17.0, 5.0],
                    [23.5, 9.5, 5.0],
                    [26.0, 2.0, 5.0]
                ]]
            );
        }

        #[test]
        fn split_tri_test() {
            let t = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 1.0, 1.0]];

            let x = split_tri(t.clone(), -1.0);
            assert_eq!(x, None);
            let x = split_tri(t.clone(), 2.0);
            assert_eq!(x, None);
            let x = split_tri(t.clone(), 0.0);
            assert_eq!(x, None);
            let x = split_tri(t.clone(), 1.0);
            assert_eq!(x, Some([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 1.0, 1.0]]));
            let x = split_tri(t.clone(), 0.7);
            assert_eq!(x, Some([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 1.0, 1.0]]));
            let x = split_tri(t.clone(), 0.2);
            assert_eq!(x, Some([[0.0, 1.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]));
            let x = split_tri(t.clone(), 0.5);
            assert_eq!(x, Some([[0.0, 1.0, 1.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]]));

            let t = [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 1.0, 0.5]];

            let x = split_tri(t.clone(), 0.5);
            assert_eq!(x, None);
        }

        #[test]
        fn split_tri_test2() {
            let t = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.5]];
            let x = split_tri(t.clone(), 0.7);
            assert_eq!(x, Some([[0.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]));
        }
        #[test]
        fn point_at_rl_test() {
            let p = point_at_rl([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0.5);
            assert_eq!(p, [0.0, 0.0, 0.5]);

            let p = point_at_rl([0.0, 0.0, 0.0], [10.0, 10.0, 1.0], 0.5);
            assert_eq!(p, [5.0, 5.0, 0.5]);

            let p = point_at_rl([10.0, 10.0, 1.0], [0.0, 0.0, 0.0], 0.5);
            assert_eq!(p, [5.0, 5.0, 0.5]);

            let p = point_at_rl([10.0, 10.0, 1.0], [0.0, 0.0, 0.0], 0.8);
            assert_eq!(p, [8.0, 8.0, 0.8]);

            let p = point_at_rl([10.0, 10.0, 0.0], [0.0, 0.0, 1.0], 0.8);
            assert_eq!(p, [2.0, 2.0, 0.8]);
        }

        #[test]
        fn contour_tri_test() {
            let t = [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
            let x = contour_tri(t, 0.5);
            assert_eq!(x, Some(([0.5, 0.5, 0.5], [0.0, 0.5, 0.5])));
        }

        #[test]
        fn flatten_edges_test() {
            let x = flatten_edges(&[(1, 2), (2, 4), (4, 0), (0, 1)]).collect::<Vec<_>>();
            assert_eq!(&x, &[1, 2, 4, 0, 1]);

            let x = flatten_edges(&[(1, 2)]).collect::<Vec<_>>();
            assert_eq!(&x, &[1, 2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_grid() {
        let g = dummy_grid();
        let t = TriMesh::from(&g);
        dbg!(&t);

        let pts = vec![
            // grid points
            [0.0, 0.0, 1.0],   // 0
            [15.0, 0.0, 2.0],  // 1
            [0.0, 15.0, 3.0],  // 2
            [15.0, 15.0, 4.0], // 3
            [0.0, 30.0, 5.0],  // 4
            [15.0, 30.0, 6.0], // 5
            // centre points
            [7.5, 7.5, 2.5],  // 6
            [7.5, 22.5, 4.5], // 7
        ];
        assert_eq!(t.points, pts);

        let tris = vec![
            (0, 6, 1),
            (1, 6, 3),
            (3, 6, 2),
            (2, 6, 0), // cell one
            (2, 7, 3),
            (3, 7, 5),
            (5, 7, 4),
            (4, 7, 2), // cell two
        ];
        assert_eq!(t.triangles, tris);
    }

    #[test]
    fn from_grid2() {
        // with nulls
        let mut g = dummy_grid();
        // remove bottom left and top right points
        g.set(0, 0, None);
        g.set(1, 2, None);

        let t = TriMesh::from(&g);
        dbg!(&t);

        let pts = vec![
            // grid points
            [15.0, 0.0, 2.0],  // 0
            [0.0, 15.0, 3.0],  // 1
            [15.0, 15.0, 4.0], // 2
            [0.0, 30.0, 5.0],  // 3
            // centre points
            [7.5, 7.5, 2.5],  // 4
            [7.5, 22.5, 4.5], // 5
        ];
        assert_eq!(t.points, pts);

        let tris = vec![
            (0, 4, 2),
            (2, 4, 1), // cell one
            (1, 5, 2),
            (3, 5, 1), // cell two
        ];
        assert_eq!(t.triangles, tris);
    }

    #[test]
    fn outlines1() {
        let g = dummy_grid();
        let t = TriMesh::from(&g);

        let x = t.outlines();

        let exp = vec![vec![
            [0.0, 30.0, 5.0],
            [0.0, 15.0, 3.0],
            [0.0, 0.0, 1.0],
            [15.0, 0.0, 2.0],
            [15.0, 15.0, 4.0],
            [15.0, 30.0, 6.0],
        ]];

        assert_eq!(x, exp);
    }

    #[test]
    fn outlines3() {
        let mut g = Grid::new([0.0, 0.0], 2, 5, 15.0);
        for (z, (x, y)) in (0..5).flat_map(|y| (0..2).map(move |x| (x, y))).enumerate() {
            g.set(x, y, z as f64);
        }
        g.set(0, 2, None);
        g.set(1, 2, None);
        let t = TriMesh::from(&g);

        let x = t.outlines();

        let exp = vec![
            vec![
                [0.0, 15.0, 2.0],
                [0.0, 0.0, 0.0],
                [15.0, 0.0, 1.0],
                [15.0, 15.0, 3.0],
            ],
            vec![
                [15.0, 45.0, 7.0],
                [0.0, 45.0, 6.0],
                [0.0, 60.0, 8.0],
                [15.0, 60.0, 9.0],
            ],
        ];

        assert_eq!(x, exp);
    }

    #[test]
    fn aabb() {
        let t = TriMesh::from(&dummy_grid());
        assert_eq!(
            t.aabb(),
            Extents3 {
                origin: [0.0, 0.0, 1.0],
                size: [15.0, 30.0, 5.0],
            }
        );
    }

    #[test]
    fn sampling_trimesh() {
        let t = TriMesh::from(&dummy_grid());
        let g = Grid::sample(&t, 6.0, true);
        let pts = g.points().map(|x| x.p3()).collect::<Vec<_>>();
        let exp = vec![
            [0.0, 0.0, 1.0],
            [6.0, 0.0, 1.4],
            [12.0, 0.0, 1.8],
            [0.0, 6.0, 1.8],
            [6.0, 6.0, 2.2],
            [12.0, 6.0, 2.6],
            [0.0, 12.0, 2.6],
            [6.0, 12.0, 3.0],
            [12.0, 12.0, 3.4],
            [0.0, 18.0, 3.4],
            [6.0, 18.0, 3.8],
            [12.0, 18.0, 4.2],
            [0.0, 24.0, 4.2],
            [6.0, 24.0, 4.6],
            [12.0, 24.0, 5.0],
        ];
        assert_eq!(pts, exp);
    }

    #[test]
    fn sampling_trimesh2() {
        let t = TriMesh::from(&dummy_grid());
        let g = Grid::sample(&t, 7.5, true);
        let pts = g.points().map(|x| x.p3()).collect::<Vec<_>>();
        let exp = vec![
            [0.0, 0.0, 1.0],
            [7.5, 0.0, 1.5],
            [0.0, 7.5, 2.0],
            [7.5, 7.5, 2.5],
            [0.0, 15.0, 3.0],
            [7.5, 15.0, 3.5],
            [0.0, 22.5, 4.0],
            [7.5, 22.5, 4.5],
        ];
        assert_eq!(pts, exp);
    }

    #[test]
    fn test_consolidation() {
        let mut mesh = TriMesh {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
                [3.0, 2.0, 3.0],
                [5.0, 4.0, 4.0],
                [6.0, 7.0, 5.0],
            ],
            triangles: vec![(1, 2, 3), (2, 3, 5)],
        };

        mesh.consolidate();
        assert_eq!(
            &mesh.points,
            &[
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 2.0],
                [3.0, 2.0, 3.0],
                [6.0, 7.0, 5.0],
            ]
        );
        assert_eq!(&mesh.triangles, &[(0, 1, 2), (1, 2, 3)]);
    }
}
