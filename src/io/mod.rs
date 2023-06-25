//! Geometry data structure module.
//! Adds functionality for data interop and conversion.
use crate::*;
use byteorder::*;
use std::{
    error::Error,
    io::{Cursor, Seek, SeekFrom::Current as Rel, Write},
};

pub mod grid;
pub mod polygon2;
pub mod polyline;
pub mod trimesh;

/// This is the decoder for Vulcan 8+ triangulations.
mod vulcan8;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn to_dxf_point(p: Point3) -> ::dxf::Point {
    let [x, y, z] = p;
    ::dxf::Point { x, y, z }
}
fn from_dxf_point(p: &::dxf::Point) -> Point3 {
    [p.x, p.y, p.z]
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
