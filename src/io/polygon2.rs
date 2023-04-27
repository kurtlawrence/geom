use super::*;

pub fn from_dxf(dxf: &[u8]) -> Result<Vec<Polygon2>> {
    let dxf =
        ::dxf::Drawing::load(&mut Cursor::new(dxf)).map_err(|e| format!("{:?} ==> {}", e, e))?;

    fn map_lw_polyline(vertices: &[::dxf::LwPolylineVertex]) -> Option<Polygon2> {
        Polygon2::new(vertices.iter().map(|v| [v.x, v.y])).ok()
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

pub fn to_dxf(polygon: &Polygon2, z: f64) -> Vec<u8> {
    let mut d = ::dxf::Drawing::new();

    let polyline = to_polyline(&mut d, polygon.iter().map(|p| p.with_z(z)));

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
                |(_, x, y, _, _)| [x, y],
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
                ps.push(Polygon2::new(points.into_iter()).map_err(|_| nom::Err::Failure(()))?);
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
