//! Data structure module.
//! Adds functionality for data interop and conversion.
use super::*;
use byteorder::*;
use std::{
    error::Error,
    io::{Cursor, Seek, SeekFrom::Current as Rel, Write},
};

/// This is the decoder for Vulcan 8+ triangulations.
mod vulcan8;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn to_dxf_point(p: Point3) -> ::dxf::Point {
    ::dxf::Point {
        x: p.0,
        y: p.1,
        z: p.2,
    }
}
fn from_dxf_point(p: &::dxf::Point) -> Point3 {
    (p.x, p.y, p.z)
}

fn to_polyline(
    drawing: &mut ::dxf::Drawing,
    ps: impl Iterator<Item = Point3>,
) -> ::dxf::entities::Polyline {
    let mut polyline = ::dxf::entities::Polyline::default();

    let vertices = ps.map(to_dxf_point).map(::dxf::entities::Vertex::new);

    for vertex in vertices {
        polyline.add_vertex(drawing, vertex);
    }

    polyline.set_is_3d_polyline(true);

    polyline
}

// ###### GRID ################################################################
impl Grid {
    // vulcan grids are determined to serialize as follows:
    // **big endian encoding**
    // 1 byte -- 0x0F
    // 31 bytes header -- fill with 0x20
    // 8 bytes f64 -- x min
    // 8 bytes f64 -- x max
    // 8 bytes f64 -- y min
    // 8 bytes f64 -- y max
    // 4 bytes u32 -- point count
    // 4 bytes padding
    // 4 bytes f32 -- x spacing
    // 4 bytes f32 -- y spacing
    // 4 bytes u32 -- x count
    // 4 bytes u32 -- y count
    // (4 bytes f32: z, 4 bytes bool: visible)
    // --> repeats for all z, row first
    pub fn from_vulcan_grid(sfg: &[u8]) -> Result<Self> {
        let mut c = Cursor::new(sfg);

        if c.read_u8()? != 0x0f {
            return Err("expecting the leading byte to be 0x0f".into());
        }

        c.seek(Rel(31))?; // skip 31 header bytes

        let xmin = c.read_f64::<BE>()?;
        let _ = c.read_f64::<BE>()?; // skip xmax
        let ymin = c.read_f64::<BE>()?;
        c.seek(Rel(16))?; // skip y max, point count, padding
        let xspace = c.read_f32::<BE>()?;
        let yspace = c.read_f32::<BE>()?;

        if (xspace - yspace).abs() > 1e-7 {
            return Err("X spacing and Y spacing must be the same".into());
        }

        let xcount = c.read_u32::<BE>()?;
        let ycount = c.read_u32::<BE>()?;

        let (xcount, ycount) = (xcount as usize, ycount as usize);

        let mut grid = Grid::new((xmin, ymin), xcount, ycount, xspace as f64);

        // points are in _row_ order; x increasing first
        let pts = (0..ycount).flat_map(|y| (0..xcount).map(move |x| (x, y)));
        for (x, y) in pts {
            let z = c.read_f32::<BE>()?;
            let visible = c.read_u32::<BE>()? == 1;
            grid.set(x, y, visible.then(|| z as f64));
        }

        Ok(grid)
    }

    pub fn to_vulcan_grid(&self) -> Vec<u8> {
        fn ser(grid: &Grid) -> Result<Vec<u8>> {
            let mut wtr = Vec::new();
            wtr.write_u8(0x0f)?; // leading byte
            wtr.write_all(&[0x20; 31])?; // header

            let max = grid
                .origin
                .add((grid.x_count() as f64, grid.y_count() as f64).scale(grid.spacing));
            wtr.write_f64::<BE>(grid.origin.0)?;
            wtr.write_f64::<BE>(max.0)?;
            wtr.write_f64::<BE>(grid.origin.1)?;
            wtr.write_f64::<BE>(max.1)?;

            wtr.write_u32::<BE>(grid.zs.len() as u32)?;
            wtr.write_all(&[0x00; 4])?;

            wtr.write_f32::<BE>(grid.spacing as f32)?;
            wtr.write_f32::<BE>(grid.spacing as f32)?;

            wtr.write_u32::<BE>(grid.x_count() as u32)?;
            wtr.write_u32::<BE>(grid.y_count() as u32)?;

            for (x, y) in (0..grid.y_count()).flat_map(|y| (0..grid.x_count()).map(move |x| (x, y)))
            {
                match grid.get(x, y) {
                    Some(g) => {
                        wtr.write_f32::<BE>(*g.z as f32)?;
                        wtr.write_u32::<BE>(1)?;
                    }
                    None => wtr.write_all(&[0x00; 8])?, // 8 bytes of zeros
                }
            }

            Ok(wtr)
        }

        ser(self).expect("serialization should not fail since writing to a memory buffer")
    }
}

// ###### POLYGON #############################################################
impl Polygon2 {
    pub fn from_dxf(dxf: &[u8]) -> Result<Vec<Polygon2>> {
        let dxf = ::dxf::Drawing::load(&mut Cursor::new(dxf))
            .map_err(|e| format!("{:?} ==> {}", e, e))?;

        fn map_lw_polyline(vertices: &[::dxf::LwPolylineVertex]) -> Option<Polygon2> {
            Polygon2::new(vertices.iter().map(|v| (v.x, v.y))).ok()
        }
        fn map_polyline<'a>(
            vertices: impl Iterator<Item = &'a ::dxf::entities::Vertex>,
        ) -> Option<Polygon2> {
            Polygon2::new(
                vertices
                    .into_iter()
                    .map(|v| from_dxf_point(&v.location).to_p2()),
            )
            .ok()
        }

        Ok(dxf
            .entities()
            .into_iter()
            .filter_map(|e| {
                // this is how to get the Z level since LwPolyline only has 2D points
                match &e.specific {
                    ::dxf::entities::EntityType::LwPolyline(p) => map_lw_polyline(&p.vertices),
                    ::dxf::entities::EntityType::Polyline(p) => map_polyline(p.vertices()),
                    _ => None,
                }
            })
            .collect())
    }

    pub fn to_dxf(&self, z: f64) -> Vec<u8> {
        let mut d = ::dxf::Drawing::new();

        let polyline = to_polyline(&mut d, self.0.iter().map(|(x, y)| (*x, *y, z)));

        d.add_entity(::dxf::entities::Entity::new(
            ::dxf::entities::EntityType::Polyline(polyline),
        ));

        d.normalize();
        let mut buf = Vec::new();
        d.save(&mut buf).expect("writing dxf to memory buffer");

        buf
    }

    pub fn from_archd(archd: &[u8]) -> Result<Vec<Polygon2>> {
        fn de(s: &str) -> std::result::Result<Vec<Polygon2>, nom::Err<()>> {
            use nom::{
                bytes::complete::*, character::complete::*, combinator::*, multi::*,
                number::complete::*, sequence::*, IResult, Parser,
            };
            // define the parsers
            fn ws<'a, G, O>(g: G) -> impl FnMut(&'a str) -> IResult<&'a str, O, ()>
            where
                G: Parser<&'a str, O, ()>,
            {
                preceded(multispace0, g)
            }
            let n = |i| ws(double)(i);
            let point = |i| {
                map(
                    preceded(ws(tag("Point:")), tuple((n, n, n, n, n))),
                    |(_, x, y, _, _)| (x, y),
                )(i)
            };

            let (s, _) = tag("FMT_3")(s)?;
            let (s, layer_name) = ws(preceded(tag("Layer:"), ws(take_till(|c| c == '\n'))))(s)?;
            let (s, _) = multispace0(s)?;

            let layer_name = layer_name.trim().strip_suffix(" 0").unwrap_or(layer_name);

            let end_layer = &*format!("End:    {}", layer_name.trim());
            let mut i = s;

            let mut ps = Vec::new();

            while !i.starts_with(end_layer) {
                let (s, polyhed) = ws(preceded(tag("POLHED:"), ws(take_till(|c| c == '\n'))))(i)?;

                let polyhed = polyhed
                    .trim()
                    .strip_suffix("0.000")
                    .unwrap_or(polyhed)
                    .trim();
                let end_seq = &*format!("End:    {}", polyhed);

                let (s, _) = take_until("Point:")(s)?;
                let (s, points) = many1(point)(s)?;
                if points.len() >= 3 {
                    ps.push(Polygon2(points));
                }

                let (s, _) = ws(tag(end_seq))(s)?;

                let (s, _) = multispace0(s)?;
                i = s;
            }

            Ok(ps)
        }

        let archd = std::str::from_utf8(archd).map_err(|_| "expecting file as utf8 string")?;
        de(archd).map_err(|_| "malformed archive file".into())
    }
}

// ###### POLYLINES ###########################################################
pub fn polylines_to_dxf<L, I, P>(layer_name: L, polylines: I) -> Vec<u8>
where
    L: std::fmt::Display,
    I: Iterator<Item = P>,
    P: AsRef<[Point3]>,
{
    use ::dxf::entities::*;

    let mut d = ::dxf::Drawing::new();

    let layer_name = layer_name.to_string();
    for polyline in polylines {
        let polyline = to_polyline(&mut d, polyline.as_ref().iter().copied());
        let mut entity = Entity::new(EntityType::Polyline(polyline));
        entity.common.layer = layer_name.clone();
        d.add_entity(entity);
    }

    d.normalize();
    let mut buf = Vec::new();
    d.save(&mut buf).expect("writing dxf to memory buffer");

    buf
}

// ###### TRIMESH #############################################################
impl TriMesh {
    // vulcan triangulations are determined to serialize as follows:
    // **big endian encoding**
    // 32 bytes padding
    // 4 bytes u32 -- 256u32 ?? // not true, not sure what this number is
    // 36 bytes padding
    // 4 bytes u32 -- point count ??
    // 20 bytes padding
    // 4 bytes u32 -- triangle count ??
    // 20 bytes padding
    // (8 bytes f64: x, 8 bytes f64: y, 8 bytes f64: z)
    // --> repeats for points
    // (4 bytes u32: p1, 4 bytes u32: p2, 4 bytes u32: p3, 12 bytes padding)
    // --> repeat for triangles
    // **Note that point indices are 1-based**
    /// Deserialize a vulcan triangulation.
    ///
    /// **Note that this _does_ support the new triangulation versions which use some weird
    /// obfuscating technique on the serialization.**
    pub fn from_vulcan_00t(tri: &[u8]) -> Result<Self> {
        // try decoding a compressed vulcan8 file
        let decoded = vulcan8::unpack_vulcan_8_00t(tri);
        // use the decoded data, or just what was supplied
        let tri = decoded.as_ref().map(|x| x.as_slice()).unwrap_or(tri);

        let mut c = Cursor::new(tri);

        c.seek(Rel(32))?; // skip 32 bytes padding

        {
            let n = c.read_u32::<BE>()?;
            log::debug!("read the integer flag, has value: {}", n);
            // return Err("expecting an integer flag 256".into());
        }

        c.seek(Rel(36))?; // skip 36 bytes padding

        let pcount = c.read_u32::<BE>()? as usize;
        c.seek(Rel(20))?; // skip 20 bytes padding
        let tcount = c.read_u32::<BE>()? as usize;
        c.seek(Rel(20))?; // skip 20 bytes padding

        let mut points = Vec::with_capacity(pcount);
        for _ in 0..pcount {
            let x = c.read_f64::<BE>()?;
            let y = c.read_f64::<BE>()?;
            let z = c.read_f64::<BE>()?;
            points.push((x, y, z));
        }

        let mut triangles = Vec::with_capacity(tcount);
        for _ in 0..tcount {
            let p1 = c.read_u32::<BE>()?.saturating_sub(1);
            let p2 = c.read_u32::<BE>()?.saturating_sub(1);
            let p3 = c.read_u32::<BE>()?.saturating_sub(1);
            c.seek(Rel(12))?;
            triangles.push((p1, p2, p3));
        }

        let mut t = Self { points, triangles };

        t.consolidate();

        Ok(t)
    }

    pub fn to_vulcan_00t(&self) -> Vec<u8> {
        fn ser(mesh: &TriMesh) -> Result<Vec<u8>> {
            let mut wtr = Vec::new();
            wtr.write_all(&[0x00; 32])?; // padding
            wtr.write_u32::<BE>(256)?;
            wtr.write_all(&[0x00; 36])?; // padding

            wtr.write_u32::<BE>(mesh.points.len() as u32)?;

            wtr.write_all(&[0x00; 20])?; // padding

            wtr.write_u32::<BE>(mesh.triangles.len() as u32)?;

            wtr.write_all(&[0x00; 20])?; // padding

            for &(x, y, z) in &mesh.points {
                wtr.write_f64::<BE>(x)?;
                wtr.write_f64::<BE>(y)?;
                wtr.write_f64::<BE>(z)?;
            }

            for &(a, b, c) in &mesh.triangles {
                wtr.write_u32::<BE>(a + 1)?;
                wtr.write_u32::<BE>(b + 1)?;
                wtr.write_u32::<BE>(c + 1)?;
                wtr.write_all(&[0x00; 12])?; // padding
            }

            Ok(wtr)
        }

        ser(self).expect("serialization should not fail since writing to a memory buffer")
    }

    // dxf triangulations use the Face3D entity
    // Face3D consists of 4 corners, so the 4th point == 1st point :shrug:
    pub fn from_dxf(dxf: &[u8]) -> Result<Self> {
        let dxf = ::dxf::Drawing::load(&mut Cursor::new(dxf))
            .map_err(|e| format!("{:?} ==> {}", e, e))?;

        collect_dxf_face3ds(&dxf)
            .or_else(|| collect_dxf_polyline_polyface_mesh(&dxf))
            .ok_or_else(|| {
                "DXF does not contain any Face3D or Polyline Polygon Mesh entities".into()
            })
            .map(|mut x| {
                x.consolidate();
                x
            })
    }

    pub fn to_dxf(&self) -> Vec<u8> {
        #[allow(clippy::boxed_local)]
        fn to_face_3d(tri: Tri) -> ::dxf::entities::Face3D {
            ::dxf::entities::Face3D {
                first_corner: to_dxf_point(tri.0),
                second_corner: to_dxf_point(tri.1),
                third_corner: to_dxf_point(tri.2),
                fourth_corner: to_dxf_point(tri.0),
                ..Default::default()
            }
        }

        let mut d = ::dxf::Drawing::new();

        let entities = self
            .tris()
            .map(to_face_3d)
            .map(::dxf::entities::EntityType::Face3D)
            .map(::dxf::entities::Entity::new);

        for entity in entities {
            d.add_entity(entity);
        }

        d.normalize();
        let mut buf = Vec::new();
        d.save(&mut buf).expect("writing dxf to memory buffer");

        buf
    }
}

fn collect_dxf_face3ds(drawing: &::dxf::Drawing) -> Option<TriMesh> {
    let mesh = drawing
        .entities()
        .into_iter()
        .filter_map(|e| match &e.specific {
            ::dxf::entities::EntityType::Face3D(f) => Some(f),
            _ => None,
        })
        .map(
            |::dxf::entities::Face3D {
                 first_corner: a,
                 second_corner: b,
                 third_corner: c,
                 ..
             }| { Tri::new((from_dxf_point(a), from_dxf_point(b), from_dxf_point(c))) },
        )
        .collect::<TriMesh>();

    (mesh.tri_len() != 0).then(|| mesh)
}

fn collect_dxf_polyline_polyface_mesh(drawing: &::dxf::Drawing) -> Option<TriMesh> {
    // <http://docs.autodesk.com/ACD/2011/ENU/filesDXF/WS1a9193826455f5ff18cb41610ec0a2e719-79c8.htm>
    //
    // This is a bit wild =/
    //
    // For polyface meshes, they are defined under the Polyline entity.
    // The polyline must have the 128 bit flag set (.get_is_polyface_mesh()).
    //
    // The autodesk has the following points about the vertices:
    // - If the vertex actually has the _location_, the 64 bit is set (as well as the 128)
    // - The vertex index values are ordered by which the vertices _appear_, the **first being
    // numbered one**
    // - If it is just a 128 flag, the 1-indexed vertices are specified
    // - if vertex is negative it means something but unrelated for our use

    fn collect_polyface_verts<'a>(
        verts: impl Iterator<Item = &'a ::dxf::entities::Vertex>,
    ) -> impl Iterator<Item = Tri> {
        let mut pts = Vec::new();
        let mut tris = Vec::new();

        for v in verts {
            if v.flags == 128 + 64 {
                // the 64bit flag is set, meaning this is vertex vertex
                pts.push(from_dxf_point(&v.location));
            } else {
                let (a, b, c) = (
                    v.polyface_mesh_vertex_index1.abs(),
                    v.polyface_mesh_vertex_index2.abs(),
                    v.polyface_mesh_vertex_index3.abs(),
                );
                if a == 0 || b == 0 || c == 0 {
                    continue;
                }
                // this vertex defines a face, 1-index so saturating sub
                tris.push((
                    (a as usize).saturating_sub(1),
                    (b as usize).saturating_sub(1),
                    (c as usize).saturating_sub(1),
                ));
            }
        }

        tris.into_iter()
            .map(move |(a, b, c)| (pts[a], pts[b], pts[c]))
            .map(Tri::new)
    }

    let mesh = drawing
        .entities()
        .into_iter()
        .filter_map(|e| match &e.specific {
            ::dxf::entities::EntityType::Polyline(p) => Some(p),
            _ => None,
        })
        .flat_map(|p| collect_polyface_verts(p.vertices()))
        .collect::<TriMesh>();

    (mesh.tri_len() != 0).then(|| mesh)
}
