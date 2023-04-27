// Tests the interop of data to geom structures
use geom::io::*;
use geom::*;

fn read(file: &str) -> Vec<u8> {
    std::fs::read(std::path::Path::new("test").join(file)).unwrap()
}

fn same(p1: Point3, p2: Point3) -> bool {
    p1.add(p2.scale(-1.0)).mag() < 1e-11
}

fn do_round_trip_test<T, D, S, Ed>(file: &str, deserialize: D, serialize: S)
where
    D: Fn(&[u8]) -> Result<T, Ed>,
    S: Fn(&T) -> Vec<u8>,
    Ed: std::fmt::Debug,
    T: PartialEq + std::fmt::Debug,
{
    let i = deserialize(&read(file)).unwrap();
    let ser = serialize(&i);
    let de = deserialize(&ser).unwrap();
    assert_eq!(i, de);
}

// Test the importing and exporting of 'out.sfg' -- a vulcan grid.
// 'out.sfg' was built with:
// - origin: (100, 200)
// - spacing: 1.0
// - zs: (null point in 3,3)
// 12 13 14
//  8  9 10 11
//  4  5  6  7
//  0  1  2  3
#[test]
fn test_out_sfg_import() {
    let g = grid::from_vulcan_grid(&read("out.sfg")).unwrap();

    assert_eq!(g.x_count(), 4);
    assert_eq!(g.y_count(), 4);

    let pts = g.points().map(|x| x.p3()).collect::<Vec<_>>();

    assert_eq!(pts.len(), 15); // missing one

    assert!(same(pts[0], [100.0, 200.0, 0.0]));
    assert!(same(pts[1], [101.0, 200.0, 1.0]));
    assert!(same(pts[2], [102.0, 200.0, 2.0]));
    assert!(same(pts[3], [103.0, 200.0, 3.0]));

    assert!(same(pts[4], [100.0, 201.0, 4.0]));
    assert!(same(pts[5], [101.0, 201.0, 5.0]));
    assert!(same(pts[6], [102.0, 201.0, 6.0]));
    assert!(same(pts[7], [103.0, 201.0, 7.0]));

    assert!(same(pts[8], [100.0, 202.0, 8.0]));
    assert!(same(pts[9], [101.0, 202.0, 9.0]));
    assert!(same(pts[10], [102.0, 202.0, 10.0]));
    assert!(same(pts[11], [103.0, 202.0, 11.0]));

    assert!(same(pts[12], [100.0, 203.0, 12.0]));
    assert!(same(pts[13], [101.0, 203.0, 13.0]));
    assert!(same(pts[14], [102.0, 203.0, 14.0]));

    assert!(g.get(3, 3).is_none());
}

#[test]
fn test_out_sfg_export() {
    do_round_trip_test("out.sfg", grid::from_vulcan_grid, grid::to_vulcan_grid);
}

// Test the importing and exporting of 'out2.sfg' -- a vulcan grid.
// 'out2.sfg' was built with:
// - origin: (100, 200)
// - spacing: 1.0
// - zs:
//  4  5  6  7
//  0  1  2  3
#[test]
fn test_out2_sfg_import() {
    let g = grid::from_vulcan_grid(&read("out2.sfg")).unwrap();

    assert_eq!(g.x_count(), 4);
    assert_eq!(g.y_count(), 2);

    let pts = g.points().map(|x| x.p3()).collect::<Vec<_>>();

    assert_eq!(pts.len(), 8);

    assert!(same(pts[0], [100.0, 200.0, 0.0]));
    assert!(same(pts[1], [101.0, 200.0, 1.0]));
    assert!(same(pts[2], [102.0, 200.0, 2.0]));
    assert!(same(pts[3], [103.0, 200.0, 3.0]));

    assert!(same(pts[4], [100.0, 201.0, 4.0]));
    assert!(same(pts[5], [101.0, 201.0, 5.0]));
    assert!(same(pts[6], [102.0, 201.0, 6.0]));
    assert!(same(pts[7], [103.0, 201.0, 7.0]));
}

#[test]
fn test_out2_sfg_export() {
    do_round_trip_test("out2.sfg", grid::from_vulcan_grid, grid::to_vulcan_grid);
}

// Test the importing and exporting of 'tri 1.00t' -- a vulcan triangulation.
// 'tri 1.00t' was built with:
// t0:
// a: (100, 200, 0)
// b: (101, 200, 1)
// c: (101, 201, 2)
#[test]
fn test_tri_1_00t_import() {
    let t = trimesh::from_vulcan_00t(&read("tri 1.00t")).unwrap();

    assert_eq!(t.tri_len(), 1);
    assert_eq!(t.point_len(), 3);

    let mut tris = t.tris().map(|x| *x);

    // t0
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 200.0, 1.0],
            [101.0, 201.0, 2.0]
        ))
    );

    assert_eq!(tris.next(), None);
}

#[test]
fn test_tri_1_00t_export() {
    do_round_trip_test(
        "tri 1.00t",
        trimesh::from_vulcan_00t,
        trimesh::to_vulcan_00t,
    );
}

// Test the importing and exporting of 'tri 2.00t' -- a vulcan triangulation.
// 'tri 2.00t' was built with:
// t0:
// a: (100, 200, 0)
// b: (101, 200, 1)
// c: (101, 201, 2)
//
// t1:
// a: (100, 200, 0)
// b: (101, 201, 2)
// c: (100, 201, 3)
#[test]
fn test_tri_2_00t_import() {
    let t = trimesh::from_vulcan_00t(&read("tri 2.00t")).unwrap();

    assert_eq!(t.tri_len(), 2);
    assert_eq!(t.point_len(), 4);

    let mut tris = t.tris().map(|x| *x);

    // t0
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 200.0, 1.0],
            [101.0, 201.0, 2.0]
        ))
    );
    // t1
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 201.0, 2.0],
            [100.0, 201.0, 3.0],
        ))
    );

    assert_eq!(tris.next(), None);
}

#[test]
fn test_tri_2_00t_export() {
    do_round_trip_test(
        "tri 2.00t",
        trimesh::from_vulcan_00t,
        trimesh::to_vulcan_00t,
    );
}

// Test the importing and exporting of 'tri 3.00t' -- a vulcan triangulation.
// 'tri 3.00t' was built with:
// t0:
// a: (100, 200, 0)
// b: (101, 200, 1)
// c: (101, 201, 2)
//
// t1:
// a: (100, 200, 0)
// b: (101, 201, 2)
// c: (100, 201, 3)
//
// t2:
// a: (104, 204, 10)
// b: (104, 205, 11)
// c: (105, 205, 12)
#[test]
fn test_tri_3_00t_import() {
    let t = trimesh::from_vulcan_00t(&read("tri 3.00t")).unwrap();

    assert_eq!(t.tri_len(), 3);
    assert_eq!(t.point_len(), 7);

    let mut tris = t.tris().map(|x| *x);

    // t0
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 200.0, 1.0],
            [101.0, 201.0, 2.0]
        ))
    );
    // t1
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 201.0, 2.0],
            [100.0, 201.0, 3.0],
        ))
    );
    // t2
    assert_eq!(
        tris.next(),
        Some((
            [104.0, 204.0, 10.0],
            [104.0, 205.0, 11.0],
            [105.0, 205.0, 12.0],
        ))
    );

    assert_eq!(tris.next(), None);
}

#[test]
fn test_tri_3_00t_export() {
    do_round_trip_test(
        "tri 3.00t",
        trimesh::from_vulcan_00t,
        trimesh::to_vulcan_00t,
    );
}

// Test the importing and exporting of 'tri 1.dxf' -- a dxf triangulation.
// 'tri 1.dxf' was built with:
// t0:
// a: (100, 200, 0)
// b: (101, 200, 1)
// c: (101, 201, 2)
#[test]
fn test_tri_1_dxf_import() {
    let t = trimesh::from_dxf(&read("tri 1.dxf")).unwrap();

    assert_eq!(t.tri_len(), 1);
    assert_eq!(t.point_len(), 3);

    let mut tris = t.tris().map(|x| *x);

    // t0
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 200.0, 1.0],
            [101.0, 201.0, 2.0]
        ))
    );

    assert_eq!(tris.next(), None);
}

#[test]
fn test_tri_1_dxf_export() {
    do_round_trip_test("tri 1.dxf", |b| trimesh::from_dxf(b), trimesh::to_dxf);
}

// Test the importing and exporting of 'tri 2.dxf' -- a dxf triangulation.
// same data as 'tri 2.00t'
#[test]
fn test_tri_2_dxf_import() {
    let t = trimesh::from_dxf(&read("tri 2.dxf")).unwrap();

    assert_eq!(t.tri_len(), 2);
    assert_eq!(t.point_len(), 4);

    let mut tris = t.tris().map(|x| *x);

    // t0
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 200.0, 1.0],
            [101.0, 201.0, 2.0]
        ))
    );
    // t1
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 201.0, 2.0],
            [100.0, 201.0, 3.0],
        ))
    );

    assert_eq!(tris.next(), None);
}

#[test]
fn test_tri_2_dxf_export() {
    do_round_trip_test("tri 2.dxf", |b| trimesh::from_dxf(b), trimesh::to_dxf);
}

// Test the importing and exporting of 'tri 3.dxf' -- a dxf triangulation.
// same data as 'tri 3.00t'
#[test]
fn test_tri_3_dxf_import() {
    let t = trimesh::from_dxf(&read("tri 3.dxf")).unwrap();

    assert_eq!(t.tri_len(), 3);
    assert_eq!(t.point_len(), 7);

    let mut tris = t.tris().map(|x| *x);

    // t0
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 200.0, 1.0],
            [101.0, 201.0, 2.0]
        ))
    );
    // t1
    assert_eq!(
        tris.next(),
        Some((
            [100.0, 200.0, 0.0],
            [101.0, 201.0, 2.0],
            [100.0, 201.0, 3.0],
        ))
    );
    // t2
    assert_eq!(
        tris.next(),
        Some((
            [104.0, 204.0, 10.0],
            [104.0, 205.0, 11.0],
            [105.0, 205.0, 12.0],
        ))
    );

    assert_eq!(tris.next(), None);
}

#[test]
fn test_tri_3_dxf_export() {
    do_round_trip_test("tri 3.dxf", |b| trimesh::from_dxf(b), trimesh::to_dxf);
}

// Test the importing of 'ipd.dxf' -- a dxf triangulation.
#[test]
fn test_ipd_dxf_import() {
    let t = trimesh::from_dxf(&read("ipd.dxf")).unwrap();

    assert_eq!(t.tri_len(), 67630);
    assert_eq!(t.point_len(), 32768);
}

// Test the importing of 'void.dxf' and 'void-face3d.dxf' -- a dxf triangulation.
#[test]
fn test_void_dxf_import() {
    let t = trimesh::from_dxf(&read("void-face3d.dxf")).unwrap();

    assert_eq!(t.tri_len(), 78884);
    assert_eq!(t.point_len(), 39904);
}

// Test the importing of 'out1.dxf' -- a shape
// s0:
// (100, 200, 0)
// (101, 200, 1)
// (101, 201, 2)
#[test]
fn test_out1_dxf_import() {
    let ps = polygon2::from_dxf(&read("out1.dxf")).unwrap();

    assert_eq!(ps.len(), 1);

    assert_eq!(
        ps[0].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );
}

#[test]
fn test_out1_dxf_export() {
    do_round_trip_test(
        "out1.dxf",
        |b| polygon2::from_dxf(b).map(|mut x| x.pop().unwrap()),
        |p| polygon2::to_dxf(p, 1.0),
    );
}

// Test the importing of 'out2.dxf' -- a shape
#[test]
fn test_out2_dxf_import() {
    let ps = polygon2::from_dxf(&read("out2.dxf")).unwrap();

    assert_eq!(ps.len(), 2);

    assert_eq!(
        ps[0].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );

    assert_eq!(
        ps[1].pts(),
        &[
            [100.0, 200.0],
            [101.0, 200.0],
            [101.0, 201.0],
            [100.0, 200.0]
        ]
    );
}

// Test the importing of 'out3.dxf' -- a shape
#[test]
fn test_out3_dxf_import() {
    let ps = polygon2::from_dxf(&read("out3.dxf")).unwrap();

    assert_eq!(ps.len(), 3);

    assert_eq!(
        ps[0].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );

    assert_eq!(
        ps[1].pts(),
        &[
            [100.0, 200.0],
            [101.0, 200.0],
            [101.0, 201.0],
            [100.0, 200.0]
        ]
    );

    assert_eq!(
        ps[2].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );
}

// Test the importing of 'combined.dxf' -- a shape
#[test]
fn test_combined_dxf_import() {
    let ps = polygon2::from_dxf(&read("combined.dxf")).unwrap();

    assert_eq!(ps.len(), 6);

    assert_eq!(
        ps[0].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );

    assert_eq!(
        ps[1].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );

    assert_eq!(
        ps[2].pts(),
        &[
            [100.0, 200.0],
            [101.0, 200.0],
            [101.0, 201.0],
            [100.0, 200.0]
        ]
    );

    assert_eq!(
        ps[3].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );

    assert_eq!(
        ps[4].pts(),
        &[
            [100.0, 200.0],
            [101.0, 200.0],
            [101.0, 201.0],
            [100.0, 200.0]
        ]
    );

    assert_eq!(
        ps[5].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );
}

// Test archd support
#[test]
fn test_out1_archd_import() {
    let ps = polygon2::from_archd(&read("out1.arch_d")).unwrap();

    assert_eq!(ps.len(), 1);

    assert_eq!(
        ps[0].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );
}

#[test]
fn test_out2_archd_import() {
    let ps = polygon2::from_archd(&read("out2.arch_d")).unwrap();

    assert_eq!(ps.len(), 2);

    assert_eq!(
        ps[0].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );

    assert_eq!(
        ps[1].pts(),
        &[
            [100.0, 200.0],
            [101.0, 200.0],
            [101.0, 201.0],
            [100.0, 200.0]
        ]
    );
}

#[test]
fn test_out3_archd_import() {
    let ps = polygon2::from_archd(&read("out3.arch_d")).unwrap();

    assert_eq!(ps.len(), 3);

    assert_eq!(
        ps[0].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );

    assert_eq!(
        ps[1].pts(),
        &[
            [100.0, 200.0],
            [101.0, 200.0],
            [101.0, 201.0],
            [100.0, 200.0]
        ]
    );

    assert_eq!(
        ps[2].pts(),
        &[[100.0, 200.0], [101.0, 200.0], [101.0, 201.0]]
    );
}

#[test]
fn test_void_mesh() {
    do_round_trip_test(
        "VOID_Simplified.00t",
        trimesh::from_vulcan_00t,
        trimesh::to_vulcan_00t,
    );
}

// Test vulcan-tries folder
#[test]
fn test_vulcan_tries_folder() {
    let t = trimesh::from_vulcan_00t(&read("vulcan-tries/spry-tri_1_blue.00t")).unwrap();
    assert_eq!(t.tri_len(), 1);
    assert_eq!(t.point_len(), 3);

    let t = trimesh::from_vulcan_00t(&read("vulcan-tries/spry-tri_1_red.00t")).unwrap();
    assert_eq!(t.tri_len(), 1);
    assert_eq!(t.point_len(), 3);

    let t = trimesh::from_vulcan_00t(&read("vulcan-tries/spry-tri_1_tri_3_sep_red.00t")).unwrap();
    assert_eq!(t.tri_len(), 2);
    assert_eq!(t.point_len(), 6);

    let t = trimesh::from_vulcan_00t(&read("vulcan-tries/spry-tri_2_blue.00t")).unwrap();
    assert_eq!(t.tri_len(), 2);
    assert_eq!(t.point_len(), 4);

    let t = trimesh::from_vulcan_00t(&read("vulcan-tries/spry-tri_2_red.00t")).unwrap();
    assert_eq!(t.tri_len(), 2);
    assert_eq!(t.point_len(), 4);

    let t = trimesh::from_vulcan_00t(&read("vulcan-tries/spry-tri_123_sep_red.00t")).unwrap();
    assert_eq!(t.tri_len(), 3);
    assert_eq!(t.point_len(), 7);
}

#[test]
fn test_trimesh_from_vulcan8() {
    let names = &[
        "tri_1_blue",
        "tri_1_red",
        "tri_1_tri_3_sep_red",
        "tri_2_blue",
        "tri_2_red",
        "tri_123_sep_red",
    ];

    for name in names {
        let t1 = &format!("vulcan-tries/spry-{}.00t", name);
        let t2 = &format!("vulcan-tries/{}.00t", name);

        println!("testing '{}'", name);

        let t1 = trimesh::from_vulcan_00t(&read(t1)).unwrap();
        let t2 = trimesh::from_vulcan_00t(&read(t2)).unwrap();
        assert_eq!(t1, t2, "vulcan 8 decoded does not match spry version");
    }
}

// Various bugs
#[test]
fn dxf_parse_errs_01() {
    let polygon = polygon2::from_dxf(&read("dxf-parse-errs/dd rs2b_1.dxf")).expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619928.0822851094, 7586047.832155782],
            [620064.7424171779, 7586181.1757720765],
            [620170.7715175577, 7586252.210161562],
            [620166.0129379041, 7586261.005382976],
            [620058.3002522367, 7586188.8242279235],
            [619920.1561541317, 7586053.929404975],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);
}

#[test]
fn dxf_parse_errs_02() {
    let polygon = polygon2::from_dxf(&read("dxf-parse-errs/dd rs2b_2.dxf")).expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619806.0073312689, 7586022.777040922],
            [619796.6303773046, 7586101.843460099],
            [619948.3626147793, 7586256.472953666],
            [620027.912899247, 7586322.142301055],
            [620104.7965473763, 7586368.231478973],
            [620099.9779893891, 7586376.993984262],
            [620036.1260203643, 7586340.1524689775],
            [619960.1582262109, 7586280.369979349],
            [619788.8668066041, 7586108.156539901],
            [619785.6270587688, 7586084.617119168],
            [619796.4869515533, 7586019.717241897],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);
}

#[test]
fn dxf_parse_errs_03() {
    let polygon = polygon2::from_dxf(&read("dxf-parse-errs/dd rs2b_3.dxf")).expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619679.0784930753, 7585970.3055708315],
            [619663.2261506754, 7586068.78081265],
            [619660.3171117192, 7586147.348047981],
            [619737.7244668179, 7586236.564606236],
            [619843.7046755889, 7586341.340473559],
            [619942.9745870918, 7586425.25533926],
            [620032.3183757473, 7586480.60239773],
            [620027.5597649124, 7586489.39760227],
            [619949.1134147939, 7586442.0562422145],
            [619861.6626125511, 7586371.389897424],
            [619735.2497353257, 7586248.457261216],
            [619651.8093951889, 7586152.651952019],
            [619652.0953720423, 7586079.535645911],
            [619669.3251814008, 7585968.098103644],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);
}

#[test]
fn dxf_parse_errs_04() {
    let polygon = polygon2::from_dxf(&read("dxf-parse-errs/dd rs2b_4.dxf")).expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619584.5090726657, 7585841.6635628175],
            [619554.9762502488, 7585916.499294599],
            [619539.8514005201, 7585985.871888512],
            [619524.8680316161, 7586188.076215829],
            [619575.1762180963, 7586256.743118009],
            [619683.5130530812, 7586372.238742269],
            [619833.1912154292, 7586509.196758727],
            [619935.7937108055, 7586580.756964459],
            [619930.5034747426, 7586589.243035541],
            [619829.2687928644, 7586518.89828553],
            [619716.6785279677, 7586418.467742554],
            [619581.2641250362, 7586278.912218699],
            [619515.6152195485, 7586192.017049341],
            [619516.0201966603, 7586104.659154818],
            [619529.9866822652, 7585984.217609994],
            [619545.436245497, 7585913.500705401],
            [619574.8870066219, 7585838.940357896],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);
}

#[test]
fn dxf_parse_errs_05() {
    let polygon = polygon2::from_dxf(&read("dxf-parse-errs/dd rs2b_5.dxf")).expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619433.7404463871, 7585710.921880401],
            [619437.647974979, 7585704.500809126],
            [619446.2745213952, 7585707.593001754],
            [619453.2596363964, 7585724.1509577],
            [619451.2559395423, 7585810.401237379],
            [619422.0257066124, 7585896.25569356],
            [619404.8447076622, 7585985.826976935],
            [619390.1044723523, 7586213.817974298],
            [619416.5665927284, 7586267.343049445],
            [619458.8578302645, 7586325.773726653],
            [619568.7698307876, 7586446.296801024],
            [619696.3918650815, 7586568.52233317],
            [619689.4071196037, 7586575.678682142],
            [619564.1213684903, 7586455.829938485],
            [619461.2007576774, 7586344.180735034],
            [619408.0943711373, 7586272.656950555],
            [619380.3323767809, 7586216.248948138],
            [619389.7027167073, 7586024.067113538],
            [619409.8580561535, 7585903.840400009],
            [619441.1619549531, 7585810.241441265],
            [619440.592547361, 7585715.882007381],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);
}

#[test]
fn malform_archd_01() {
    let polygon = polygon2::from_archd(&read("rs2_bench.arch_d")).expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619928.9, 7586048.897],
            [620066.623, 7586182.754],
            [620174.036, 7586254.03],
            [620169.168, 7586262.766],
            [620060.203, 7586190.42],
            [619921.1, 7586055.155],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);
}

#[test]
fn should_be_one_polygon() {
    let polygon = polygon2::from_dxf(&read(
        "one-polygon/1. STAGE PLAN RESHAPE_STAGE PLAN 231231_DAEDELUS_RS2.dxf",
    ))
    .expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619928.0822851094, 7586047.832155782],
            [620064.7424171779, 7586181.1757720765],
            [620170.7715175577, 7586252.210161562],
            [620166.0129379041, 7586261.005382976],
            [620058.3002522367, 7586188.8242279235],
            [619920.1561541317, 7586053.929404975],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);

    let polygon = polygon2::from_dxf(&read(
        "one-polygon/2. STAGE PLAN RESHAPE_STAGE PLAN 231231_DAEDELUS_RS2.dxf",
    ))
    .expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619806.0073312689, 7586022.777040922],
            [619796.6303773046, 7586101.843460099],
            [619948.3626147793, 7586256.472953666],
            [620027.912899247, 7586322.142301055],
            [620104.7965473763, 7586368.231478973],
            [620099.9779893891, 7586376.993984262],
            [620036.1260203643, 7586340.1524689775],
            [619960.1582262109, 7586280.369979349],
            [619788.8668066041, 7586108.156539901],
            [619785.6270587688, 7586084.617119168],
            [619796.4869515533, 7586019.717241897],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);

    let polygon = polygon2::from_dxf(&read(
        "one-polygon/3. STAGE PLAN RESHAPE_STAGE PLAN 231231_DAEDELUS_RS2.dxf",
    ))
    .expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619679.0784930753, 7585970.3055708315],
            [619663.2261506754, 7586068.78081265],
            [619660.3171117192, 7586147.348047981],
            [619737.7244668179, 7586236.564606236],
            [619843.7046755889, 7586341.340473559],
            [619942.9745870918, 7586425.25533926],
            [620032.3183757473, 7586480.60239773],
            [620027.5597649124, 7586489.39760227],
            [619949.1134147939, 7586442.0562422145],
            [619861.6626125511, 7586371.389897424],
            [619735.2497353257, 7586248.457261216],
            [619651.8093951889, 7586152.651952019],
            [619652.0953720423, 7586079.535645911],
            [619669.3251814008, 7585968.098103644],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);

    let polygon = polygon2::from_dxf(&read(
        "one-polygon/4. STAGE PLAN RESHAPE_STAGE PLAN 231231_DAEDELUS_RS2.dxf",
    ))
    .expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619584.5090726657, 7585841.6635628175],
            [619554.9762502488, 7585916.499294599],
            [619539.8514005201, 7585985.871888512],
            [619524.8680316161, 7586188.076215829],
            [619575.1762180963, 7586256.743118009],
            [619683.5130530812, 7586372.238742269],
            [619833.1912154292, 7586509.196758727],
            [619935.7937108055, 7586580.756964459],
            [619930.5034747426, 7586589.243035541],
            [619829.2687928644, 7586518.89828553],
            [619716.6785279677, 7586418.467742554],
            [619581.2641250362, 7586278.912218699],
            [619515.6152195485, 7586192.017049341],
            [619516.0201966603, 7586104.659154818],
            [619529.9866822652, 7585984.217609994],
            [619545.436245497, 7585913.500705401],
            [619574.8870066219, 7585838.940357896],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);

    let polygon = polygon2::from_dxf(&read(
        "one-polygon/5. STAGE PLAN RESHAPE_STAGE PLAN 231231_DAEDELUS_RS2.dxf",
    ))
    .expect("parse fine");
    let expected = vec![Polygon2::new(
        vec![
            [619433.7404463871, 7585710.921880401],
            [619437.647974979, 7585704.500809126],
            [619446.2745213952, 7585707.593001754],
            [619453.2596363964, 7585724.1509577],
            [619451.2559395423, 7585810.401237379],
            [619422.0257066124, 7585896.25569356],
            [619404.8447076622, 7585985.826976935],
            [619390.1044723523, 7586213.817974298],
            [619416.5665927284, 7586267.343049445],
            [619458.8578302645, 7586325.773726653],
            [619568.7698307876, 7586446.296801024],
            [619696.3918650815, 7586568.52233317],
            [619689.4071196037, 7586575.678682142],
            [619564.1213684903, 7586455.829938485],
            [619461.2007576774, 7586344.180735034],
            [619408.0943711373, 7586272.656950555],
            [619380.3323767809, 7586216.248948138],
            [619389.7027167073, 7586024.067113538],
            [619409.8580561535, 7585903.840400009],
            [619441.1619549531, 7585810.241441265],
            [619440.592547361, 7585715.882007381],
        ]
        .into_iter(),
    )
    .unwrap()];

    assert_eq!(polygon, expected);
}
