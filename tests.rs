use super::*;

fn dummy_grid() -> Grid {
    let mut g = Grid::new((0.0, 0.0), 2, 3, 15.0);
    (0..3)
        .flat_map(|y| (0..2).map(move |x| (x, y)))
        .zip(1..7)
        .for_each(|((x, y), z)| g.set(x, y, z as f64));
    g
}

// ###### POINT ###############################################################
#[test]
fn point_adding() {
    let p = (0.0, 1.0).add((3.0, 1.0));
    assert_eq!(p, (3.0, 2.0));

    let p = (0.0, 1.0, 5.0).add((3.0, 1.0));
    assert_eq!(p, (3.0, 2.0, 5.0));

    let p = (0.0, 1.0, 5.0).add((3.0, 1.0, 5.0));
    assert_eq!(p, (3.0, 2.0, 10.0));
}

#[test]
fn point_scaling() {
    let p = (0.0, 1.0).scale(2.0);
    assert_eq!(p, (0.0, 2.0));

    let p = (-2.0, 0.5, 3.0).scale(-0.5);
    assert_eq!(p, (1.0, -0.25, -1.5));
}

#[test]
fn to_point_testing() {
    assert_eq!((0.0, 1.0).to_p2(), (0.0, 1.0));
    assert_eq!((0.0, 1.0, 2.0).to_p2(), (0.0, 1.0));
}

#[test]
fn xproduct_test() {
    let v = xprod((1.0, 0.0, 0.0), (0.0, 1.0, 0.0));
    assert_eq!(v, (0.0, 0.0, 1.0));

    let v = xprod((1.0, 1.0, 0.0), (-1.0, 1.0, 0.0));
    assert_eq!(v, (0.0, -0.0, 2.0));
}

#[test]
fn grade_testing() {
    let x = grade((0.0, 10.0, 1.0)) - 0.1;
    assert!(x.abs() < 1e-11);

    let x = grade((0.0, 10.0, -1.0)) - -0.1;
    assert!(x.abs() < 1e-11);

    let x = grade((3.0, 4.0, 1.0)) - 0.2;
    assert!(x.abs() < 1e-11);

    let x = grade((3.0, 4.0, -1.0)) - -0.2;
    assert!(x.abs() < 1e-11);
}

#[test]
fn len_xy_testing() {
    let x = len_xy((3.0, 4.0, 1.0)) - 5.0;
    assert!(x.abs() < 1e-11);

    assert_eq!(zero_len_xy((0.0, 0.0, 1.0)), true);
    assert_eq!(zero_len_xy((0.0, 1.0, 1.0)), false);
    assert_eq!(zero_len_xy((1.0, 0.0, 1.0)), false);
    assert_eq!(zero_len_xy((1.0, 1.0, 1.0)), false);
}

#[test]
fn mag_testing() {
    let m = (3.0, 4.0).mag() - 5.0;
    assert!(m.abs() < 1e-11);

    let m = (3.0, -4.0).mag() - 5.0;
    assert!(m.abs() < 1e-11);

    let m = (-3.0, 4.0).mag() - 5.0;
    assert!(m.abs() < 1e-11);

    let m = (2.0, 3.0, 6.0).mag() - 7.0;
    assert!(m.abs() < 1e-11);

    let m = (2.0, -3.0, 6.0).mag() - 7.0;
    assert!(m.abs() < 1e-11);

    let m = (-2.0, -3.0, 6.0).mag() - 7.0;
    assert!(m.abs() < 1e-11);

    let m = (-2.0, -3.0, -6.0).mag() - 7.0;
    assert!(m.abs() < 1e-11);
}

#[test]
fn unit_vector() {
    let u = (2.0, 0.0).unit();
    assert_eq!(u, (1.0, 0.0));

    let u = (0.0, 0.0, 2.0).unit();
    assert_eq!(u, (0.0, 0.0, 1.0));
}

// ###### POLYGON #############################################################
#[test]
fn point_inside_testing() {
    let polygon = Polygon2::new(
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
            .iter()
            .copied(),
    )
    .unwrap();
    assert_eq!(polygon.inside((0.5, 0.5)), true);
    assert_eq!(polygon.inside((0.9, 0.9)), true);
    assert_eq!(polygon.inside((0.9999999, 0.999999)), true);
    assert_eq!(polygon.inside((0.00001, 0.00001)), true);
    assert_eq!(polygon.inside((0.3, 0.1)), true);

    assert_eq!(polygon.inside((-0.3, 0.1)), false);
    assert_eq!(polygon.inside((0.3, -0.1)), false);
    assert_eq!(polygon.inside((1.1, -0.1)), false);
    assert_eq!(polygon.inside((1.1, 0.1)), false);
    assert_eq!(polygon.inside((0.5, 1.1)), false);
    assert_eq!(polygon.inside((0.5, -1.1)), false);
    assert_eq!(polygon.inside((1.5, 1.1)), false);

    // notice that when on upper boundary it is false
    assert_eq!(polygon.inside((0.0, 0.0)), true);
    assert_eq!(polygon.inside((0.5, 0.0)), true);
    assert_eq!(polygon.inside((1.0, 0.0)), false);
    assert_eq!(polygon.inside((1.0, 0.5)), false);
    assert_eq!(polygon.inside((1.0, 1.0)), false);
    assert_eq!(polygon.inside((0.5, 1.0)), false);
    assert_eq!(polygon.inside((0.0, 1.0)), false);
    assert_eq!(polygon.inside((0.0, 0.5)), true);
}

// ###### GRID ################################################################
#[test]
fn new_grid() {
    let g = Grid::new((0.0, 0.0), 10, 20, 15.0);
    assert_eq!(g.origin(), (0.0, 0.0));
    assert_eq!(g.x_count(), 10);
    assert_eq!(g.y_count(), 20);
    assert_eq!(g.spacing(), 15.0);
    assert_eq!(g.len(), 200);
}

#[test]
fn get_xy_test() {
    let g = Grid::new((0.0, 0.0), 10, 20, 15.0);
    assert_eq!(g.get_xy(0, 0), (0.0, 0.0));
    assert_eq!(g.get_xy(1, 0), (15.0, 0.0));
    assert_eq!(g.get_xy(0, 1), (0.0, 15.0));
    assert_eq!(g.get_xy(3, 4), (45.0, 60.0));
}

#[test]
fn populate_grid() {
    let g = dummy_grid();

    assert_eq!(
        &g.zs,
        &[
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
            Some(6.0)
        ]
    );

    assert_eq!(g.get(0, 0).map(|p| p.p3()), Some((0.0, 0.0, 1.0)));
    assert_eq!(g.get(1, 0).map(|p| p.p3()), Some((15.0, 0.0, 2.0)));
    assert_eq!(g.get(0, 1).map(|p| p.p3()), Some((0.0, 15.0, 3.0)));
    assert_eq!(g.get(1, 1).map(|p| p.p3()), Some((15.0, 15.0, 4.0)));
    assert_eq!(g.get(0, 2).map(|p| p.p3()), Some((0.0, 30.0, 5.0)));
    assert_eq!(g.get(1, 2).map(|p| p.p3()), Some((15.0, 30.0, 6.0)));
}

#[test]
fn grid_points() {
    let g = dummy_grid();
    let mut points = g.points();

    let p = points.next().unwrap();
    assert_eq!(p.x(), 0.0);
    assert_eq!(p.x_idx(), 0);
    assert_eq!(p.y(), 0.0);
    assert_eq!(p.y_idx(), 0);
    assert_eq!(p.p2(), (0.0, 0.0));
    assert_eq!(p.p3(), (0.0, 0.0, 1.0));

    assert_eq!(
        points.next().map(|p| (p.idx(), p.p3())),
        Some((1, (15.0, 0.0, 2.0)))
    );
    assert_eq!(
        points.next().map(|p| (p.idx(), p.p3())),
        Some((2, (0.0, 15.0, 3.0)))
    );
    assert_eq!(points.next().map(|p| p.p3()), Some((15.0, 15.0, 4.0)));
    assert_eq!(points.next().map(|p| p.p3()), Some((0.0, 30.0, 5.0)));
    assert_eq!(
        points.next().map(|p| (p.idx(), p.p3())),
        Some((5, (15.0, 30.0, 6.0)))
    );
}

// ###### TRIMESH #############################################################
#[test]
fn from_grid() {
    let g = dummy_grid();
    let t = TriMesh::from(&g);
    dbg!(&t);

    let pts = vec![
        // grid points
        (0.0, 0.0, 1.0),   // 0
        (15.0, 0.0, 2.0),  // 1
        (0.0, 15.0, 3.0),  // 2
        (15.0, 15.0, 4.0), // 3
        (0.0, 30.0, 5.0),  // 4
        (15.0, 30.0, 6.0), // 5
        // centre points
        (7.5, 7.5, 2.5),  // 6
        (7.5, 22.5, 4.5), // 7
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
        (15.0, 0.0, 2.0),  // 0
        (0.0, 15.0, 3.0),  // 1
        (15.0, 15.0, 4.0), // 2
        (0.0, 30.0, 5.0),  // 3
        // centre points
        (7.5, 7.5, 2.5),  // 4
        (7.5, 22.5, 4.5), // 5
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
        (0.0, 30.0, 5.0),
        (0.0, 15.0, 3.0),
        (0.0, 0.0, 1.0),
        (15.0, 0.0, 2.0),
        (15.0, 15.0, 4.0),
        (15.0, 30.0, 6.0),
    ]];

    assert_eq!(x, exp);
}

#[test]
fn outlines3() {
    let mut g = Grid::new((0.0, 0.0), 2, 5, 15.0);
    for (z, (x, y)) in (0..5).flat_map(|y| (0..2).map(move |x| (x, y))).enumerate() {
        g.set(x, y, z as f64);
    }
    g.set(0, 2, None);
    g.set(1, 2, None);
    let t = TriMesh::from(&g);

    let x = t.outlines();

    let exp = vec![
        vec![
            (0.0, 15.0, 2.0),
            (0.0, 0.0, 0.0),
            (15.0, 0.0, 1.0),
            (15.0, 15.0, 3.0),
        ],
        vec![
            (15.0, 45.0, 7.0),
            (0.0, 45.0, 6.0),
            (0.0, 60.0, 8.0),
            (15.0, 60.0, 9.0),
        ],
    ];

    assert_eq!(x, exp);
}

#[test]
fn aabb() {
    let t = TriMesh::from(&dummy_grid());
    assert_eq!(t.aabb(), ((0.0, 0.0, 1.0), (15.0, 30.0, 6.0)));
}

#[test]
fn sampling_trimesh() {
    let t = TriMesh::from(&dummy_grid());
    let g = Grid::sample(&t, 6.0, true);
    let pts = g.points().map(|x| x.p3()).collect::<Vec<_>>();
    let exp = vec![
        (0.0, 0.0, 1.0),
        (6.0, 0.0, 1.4),
        (12.0, 0.0, 1.8),
        (0.0, 6.0, 1.8),
        (6.0, 6.0, 2.2),
        (12.0, 6.0, 2.6),
        (0.0, 12.0, 2.6),
        (6.0, 12.0, 3.0),
        (12.0, 12.0, 3.4),
        (0.0, 18.0, 3.4),
        (6.0, 18.0, 3.8),
        (12.0, 18.0, 4.2),
        (0.0, 24.0, 4.2),
        (6.0, 24.0, 4.6),
        (12.0, 24.0, 5.0),
    ];
    assert_eq!(pts, exp);
}

#[test]
fn sampling_trimesh2() {
    let t = TriMesh::from(&dummy_grid());
    let g = Grid::sample(&t, 7.5, true);
    let pts = g.points().map(|x| x.p3()).collect::<Vec<_>>();
    let exp = vec![
        (0.0, 0.0, 1.0),
        (7.5, 0.0, 1.5),
        (0.0, 7.5, 2.0),
        (7.5, 7.5, 2.5),
        (0.0, 15.0, 3.0),
        (7.5, 15.0, 3.5),
        (0.0, 22.5, 4.0),
        (7.5, 22.5, 4.5),
    ];
    assert_eq!(pts, exp);
}

#[test]
fn test_consolidation() {
    let mut mesh = TriMesh {
        points: vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 2.0),
            (3.0, 2.0, 3.0),
            (5.0, 4.0, 4.0),
            (6.0, 7.0, 5.0),
        ],
        triangles: vec![(1, 2, 3), (2, 3, 5)],
    };

    mesh.consolidate();
    assert_eq!(
        &mesh.points,
        &[
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 2.0),
            (3.0, 2.0, 3.0),
            (6.0, 7.0, 5.0),
        ]
    );
    assert_eq!(&mesh.triangles, &[(0, 1, 2), (1, 2, 3)]);
}

// ###### PLANE ###############################################################
#[test]
fn fit_least_sqs_plane() {
    // going to test a plane which pivots along (-1,1)->(1,-1) at 45° rotation
    let plane = Plane::fit_least_sqs(&[
        (-1.0, -1.0, -(2f64.sqrt())),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 2f64.sqrt()),
    ])
    .unwrap();

    dbg!(&plane);

    assert_eq!(plane.centroid(), (0.0, 0.0, 0.0));
    let normal = plane.normal;
    let unit_normal = normal
        .unit()
        .add((-0.5, -0.5, 0.5_f64.sqrt()).scale(-1.0))
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
        .add((0.5_f64.sqrt(), -(0.5_f64.sqrt()), 0.0))
        .mag();
    assert!(strike.abs() < 1e-11);

    let dip = plane.dip().unit();
    dbg!(dip);
    let dip = dip.add((0.5, 0.5, 0.5_f64.sqrt())).mag();
    assert!(dip.abs() < 1e-11);

    let x = plane.register_z((0.5, -0.5)) - 0.0;
    assert!(x.abs() < 1e-11);

    let x = plane.register_z((-0.5, 0.5)) - 0.0;
    assert!(x.abs() < 1e-11);

    let x = plane.register_z((0.5, 0.5)) - 0.5_f64.sqrt();
    assert!(x.abs() < 1e-11);

    let x = plane.register_z((-0.5, -0.5)) - -0.5_f64.sqrt();
    assert!(x.abs() < 1e-11);
}

// ###### LINE 2D #############################################################
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
    assert_eq!(line.pt_cmp((1.0, 0.0)), Equal);
    assert_eq!(line.pt_cmp((2.0, 4.0)), Less);
    assert_eq!(line.pt_cmp((2.0, 0.0)), Greater);

    // -ve gradient
    let line = Line2 {
        a: 3.0,
        b: 2.0,
        c: -4.0,
    };
    assert_eq!(line.pt_cmp((-8.0, 10.0)), Equal);
    assert_eq!(line.pt_cmp((2.0, 4.0)), Less);
    assert_eq!(line.pt_cmp((-5.0, -1.0)), Greater);

    // vertical
    let line = Line2 {
        a: 1.0,
        b: 0.0,
        c: 0.0,
    };
    assert_eq!(line.pt_cmp((0.0, 6.0)), Equal);
    assert_eq!(line.pt_cmp((1.0, 6.0)), Less);
    assert_eq!(line.pt_cmp((-0.5, 6.0)), Greater);

    // horizontal
    let line = Line2 {
        a: 0.0,
        b: 1.0,
        c: 0.0,
    };
    assert_eq!(line.pt_cmp((6.0, 0.0)), Equal);
    assert_eq!(line.pt_cmp((6.0, 1.0)), Less);
    assert_eq!(line.pt_cmp((6.0, -0.5)), Greater);
}

#[test]
fn line2_intercepts() {
    let line = Line2 {
        a: 2.0,
        b: -4.0,
        c: 8.0,
    };
    assert_eq!(line.x_intercept(), Some((4.0, 0.0)));
    assert_eq!(line.y_intercept(), Some((0.0, -2.0)));

    let vert = Line2 {
        a: 1.0,
        b: 0.0,
        c: 4.0,
    };
    assert_eq!(vert.x_intercept(), Some((4.0, 0.0)));
    assert_eq!(vert.y_intercept(), None);

    let horz = Line2 {
        a: 0.0,
        b: 2.0,
        c: 1.0,
    };
    assert_eq!(horz.x_intercept(), None);
    assert_eq!(horz.y_intercept(), Some((0.0, 0.5)));
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
