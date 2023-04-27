use criterion::*;
use geom::*;

fn polygons(c: &mut Criterion) {
    c.bench_function("point inside small polygon", |b| {
        let polygon = Polygon2::new(
            [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
                .iter()
                .copied(),
        )
        .unwrap();
        b.iter(|| polygon.envelops((0.34, 0.578)))
    });
    c.bench_function("point inside large polygon", |b| {
        let points = (0..100)
            .map(|x| (0.0, x as f64))
            .chain((0..100).map(|x| (x as f64, 100.0)))
            .chain((0..100).rev().map(|x| (100.0, x as f64)))
            .chain((1..100).rev().map(|x| (x as f64, 0.0)));
        let polygon = Polygon2::new(points).unwrap();
        b.iter(|| polygon.envelops((3.34, 78.578)))
    });
}

fn planes(c: &mut Criterion) {
    fn points(size: usize) -> Vec<Point3> {
        (0..size)
            .flat_map(move |x| (0..size).map(move |y| (x, y)))
            .enumerate()
            .map(|(z, (x, y))| (x as f64, y as f64, z as f64))
            .collect()
    }
    c.bench_function("fit least sqs small", |b| {
        let points = points(5);
        b.iter(|| Plane::fit_least_sqs(&points)) // size used in shapr
    });
    c.bench_function("fit least sqs large", |b| {
        let points = points(100);
        b.iter(|| Plane::fit_least_sqs(&points)) // 10_000 points is a lot!
    });
}

fn trimeshs(c: &mut Criterion) {
    fn grid(size: usize) -> Grid {
        let mut g = Grid::new((300.0, 150.0), size, size, 30.0);
        let p = (0..size)
            .flat_map(move |x| (0..size).map(move |y| (x, y)))
            .enumerate();
        for (z, (x, y)) in p {
            g.set(x, y, z as f64);
        }

        g
    }

    c.bench_function("outlines single", |b| {
        let mesh = TriMesh::from(&grid(100));
        b.iter(|| mesh.outlines());
    });

    c.bench_function("triangulate small", |b| {
        let grid = grid(100);
        b.iter(|| TriMesh::from(&grid))
    });

    c.bench_function("extend trimesh", |b| {
        let g = &grid(10);
        let tri = TriMesh::from(&grid(25));
        b.iter(|| TriMesh::from(g).extend(tri.tris()))
    });

    c.bench_function("trimesh aabb 10", |b| {
        let g = TriMesh::from(&grid(10));
        b.iter(|| g.aabb())
    });

    c.bench_function("trimesh aabb 100", |b| {
        let g = TriMesh::from(&grid(100));
        b.iter(|| g.aabb())
    });

    c.bench_function("trimesh aabb 250", |b| {
        let g = TriMesh::from(&grid(250));
        b.iter(|| g.aabb())
    });

    c.bench_function("contour single rl 10", |b| {
        let t = TriMesh::from(&grid(10));
        b.iter(|| t.contour_rl(5.0))
    });

    c.bench_function("contour single rl 100", |b| {
        let t = TriMesh::from(&grid(100));
        b.iter(|| t.contour_rl(25.0))
    });

    c.bench_function("contours 10 1m", |b| {
        let t = TriMesh::from(&grid(10));
        b.iter(|| t.contour(1.0))
    });

    c.bench_function("contours 100 2m", |b| {
        let t = TriMesh::from(&grid(100));
        b.iter(|| t.contour(2.0))
    });

    c.bench_function("contours 1m actual", |b| {
        let t = io::trimesh::from_vulcan_00t(&std::fs::read("test/VOID_Simplified.00t").unwrap())
            .unwrap();
        b.iter(|| t.contour(1.0))
    });
}

fn grids(c: &mut Criterion) {
    fn grid(size: usize) -> Grid {
        let mut g = Grid::new((300.0, 150.0), size, size, 30.0);
        let p = (0..size)
            .flat_map(move |x| (0..size).map(move |y| (x, y)))
            .enumerate();
        for (z, (x, y)) in p {
            g.set(x, y, z as f64);
        }

        g
    }

    c.bench_function("grid sample 10", |b| {
        let mesh = TriMesh::from(&grid(10));
        b.iter(|| Grid::sample(&mesh, 5.0, true));
    });

    c.bench_function("grid sample 100", |b| {
        let mesh = TriMesh::from(&grid(100));
        b.iter(|| Grid::sample(&mesh, 5.0, true));
    });

    c.bench_function("grid sample 250", |b| {
        let mesh = TriMesh::from(&grid(250));
        b.iter(|| Grid::sample(&mesh, 5.0, true));
    });
}

criterion_group!(benches, polygons, planes, trimeshs, grids);
criterion_main!(benches);
