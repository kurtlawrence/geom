use super::*;

pub fn to_dxf<L, I, P>(layer_name: L, polylines: I) -> Vec<u8>
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

pub fn from_dxf(dxf: &[u8]) -> Result<Vec<Polyline3>> {
    let dxf =
        ::dxf::Drawing::load(&mut Cursor::new(dxf)).map_err(|e| format!("{:?} ==> {}", e, e))?;

    fn map_polyline<'a>(
        vertices: impl Iterator<Item = &'a ::dxf::entities::Vertex>,
    ) -> Option<Polyline3> {
        Polyline3::new(vertices.into_iter().map(|v| from_dxf_point(&v.location))).ok()
    }

    Ok(dxf
        .entities()
        .into_iter()
        .filter_map(|e| {
            match &e.specific {
                ::dxf::entities::EntityType::Polyline(p) => map_polyline(p.vertices()),
                _ => None,
            }
        })
        .collect())
}
