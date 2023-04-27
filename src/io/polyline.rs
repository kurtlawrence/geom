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
