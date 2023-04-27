use super::*;

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
pub fn from_vulcan_00t(tri: &[u8]) -> Result<TriMesh> {
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
        points.push([x, y, z]);
    }

    let mut triangles = Vec::with_capacity(tcount);
    for _ in 0..tcount {
        let p1 = c.read_u32::<BE>()?.saturating_sub(1);
        let p2 = c.read_u32::<BE>()?.saturating_sub(1);
        let p3 = c.read_u32::<BE>()?.saturating_sub(1);
        c.seek(Rel(12))?;
        triangles.push((p1, p2, p3));
    }

    Ok(TriMesh::from_raw(points, triangles))
}

pub fn to_vulcan_00t(mesh: &TriMesh) -> Vec<u8> {
    fn ser(mesh: &TriMesh) -> Result<Vec<u8>> {
        let mut wtr = Vec::new();
        wtr.write_all(&[0x00; 32])?; // padding
        wtr.write_u32::<BE>(256)?;
        wtr.write_all(&[0x00; 36])?; // padding

        wtr.write_u32::<BE>(mesh.point_len() as u32)?;

        wtr.write_all(&[0x00; 20])?; // padding

        wtr.write_u32::<BE>(mesh.tri_len() as u32)?;

        wtr.write_all(&[0x00; 20])?; // padding

        for &[x, y, z] in mesh.points() {
            wtr.write_f64::<BE>(x)?;
            wtr.write_f64::<BE>(y)?;
            wtr.write_f64::<BE>(z)?;
        }

        for &(a, b, c) in mesh.tri_indices() {
            wtr.write_u32::<BE>(a + 1)?;
            wtr.write_u32::<BE>(b + 1)?;
            wtr.write_u32::<BE>(c + 1)?;
            wtr.write_all(&[0x00; 12])?; // padding
        }

        Ok(wtr)
    }

    ser(mesh).expect("serialization should not fail since writing to a memory buffer")
}

// dxf triangulations use the Face3D entity
// Face3D consists of 4 corners, so the 4th point == 1st point :shrug:
pub fn from_dxf(dxf: &[u8]) -> Result<TriMesh> {
    let dxf =
        ::dxf::Drawing::load(&mut Cursor::new(dxf)).map_err(|e| format!("{:?} ==> {}", e, e))?;

    collect_dxf_face3ds(&dxf)
        .or_else(|| collect_dxf_polyline_polyface_mesh(&dxf))
        .ok_or_else(|| "DXF does not contain any Face3D or Polyline Polygon Mesh entities".into())
        .map(|mut x| {
            x.consolidate();
            x
        })
}

pub fn to_dxf(mesh: &TriMesh) -> Vec<u8> {
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

    let entities = mesh
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
