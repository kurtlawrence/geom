use super::*;

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
pub fn from_vulcan_grid(sfg: &[u8]) -> Result<Grid> {
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

    let mut grid = Grid::new([xmin, ymin], xcount, ycount, xspace as f64);

    // points are in _row_ order; x increasing first
    let pts = (0..ycount).flat_map(|y| (0..xcount).map(move |x| (x, y)));
    for (x, y) in pts {
        let z = c.read_f32::<BE>()?;
        let visible = c.read_u32::<BE>()? == 1;
        grid.set(x, y, visible.then(|| z as f64));
    }

    Ok(grid)
}

pub fn to_vulcan_grid(grid: &Grid) -> Vec<u8> {
    fn ser(grid: &Grid) -> Result<Vec<u8>> {
        let mut wtr = Vec::new();
        wtr.write_u8(0x0f)?; // leading byte
        wtr.write_all(&[0x20; 31])?; // header

        let max = grid
            .origin()
            .add([grid.x_count() as f64, grid.y_count() as f64].scale(grid.spacing()));
        wtr.write_f64::<BE>(grid.origin()[0])?;
        wtr.write_f64::<BE>(max[0])?;
        wtr.write_f64::<BE>(grid.origin()[1])?;
        wtr.write_f64::<BE>(max[1])?;

        wtr.write_u32::<BE>(grid.len() as u32)?;
        wtr.write_all(&[0x00; 4])?;

        wtr.write_f32::<BE>(grid.spacing() as f32)?;
        wtr.write_f32::<BE>(grid.spacing() as f32)?;

        wtr.write_u32::<BE>(grid.x_count() as u32)?;
        wtr.write_u32::<BE>(grid.y_count() as u32)?;

        for (x, y) in (0..grid.y_count()).flat_map(|y| (0..grid.x_count()).map(move |x| (x, y))) {
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

    ser(grid).expect("serialization should not fail since writing to a memory buffer")
}
