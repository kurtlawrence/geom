use byteorder::*;
use std::io::{prelude::*, Cursor, Error, ErrorKind, Result, SeekFrom::*};

/// The start location of _blocks_.
/// This is populated after parsing the header data, and in reverse order so each time
/// a block is read, the location can be popped off and removed.
type BlockLocations = Vec<u64>;
/// Byte buffer alias.
type Bytes = Vec<u8>;

pub fn unpack_vulcan_8_00t(packed: &[u8]) -> Result<Bytes> {
    let mut file = Cursor::new(packed);

    let offset = parse_header(&mut file)?;

    debug_assert_eq!(file.position(), 28, "expecting to be at file position 28");

    let block_locations = populate_block_locations(&mut file, offset.into())?;

    // Extract the data spread out in the file according to block locations.
    let data = extract_blocks(block_locations, file)?;

    let output = Vec::with_capacity(data.len()); // will be at least data.len size

    State {
        bytes: data,
        output,
    }
    .unpack()
}

struct State {
    bytes: Bytes,
    output: Bytes,
}

impl State {
    fn unpack(mut self) -> Result<Bytes> {
        self.bytes.reverse(); // the algorithm repeatedly reads bytes in order, we can use data like a stack for efficiency

        // repeatedly read in literal blocks of bytes until a compression byte flag
        // is reached
        while let Some(cmpr_byte) = self.read_write_block_until_offset_byte()? {
            // determine the characteristics of the compression
            let len = self.determine_compression_len(cmpr_byte)?;
            let offset = self.determine_compression_offset(cmpr_byte)?;

            // apply the expansion starting at `offset` and going for `len`
            overwrite_within(&mut self.output, offset, len)?;
        }

        Ok(self.output)
    }

    fn determine_compression_len(&mut self, higher_ord_byte: u8) -> Result<usize> {
        // The length is stored in the 3 highest order bits
        let len = (higher_ord_byte >> 5) as usize;

        if len == 7 {
            // if len is down to 7 (that is the byte was >= 0xE0 -- 224) we
            // read the next byte and set that +7 as the new iters
            // Note that this fits the repeated pattern of E0 FF 00, and that the
            // range also covers FA FB FC FD
            let len = self.bytes.pop().ok_or_else(|| {
                Error::new(
                    ErrorKind::UnexpectedEof,
                    "trying to read past end of file when getting next block byte",
                )
            })?;

            Ok(len as usize + 7)
        } else {
            Ok(len)
        }
    }

    fn determine_compression_offset(&mut self, higher_ord_byte: u8) -> Result<usize> {
        // nonread_byte is >= 0x20 (32)
        // & with 0x1F (31) -- this is 0001 1111
        // this means the 3 highest bits are masked, which is the length
        // this makes the number be something 0..=31
        // shifting this to front end of 16 bit number makes it multiply by 256 (8 bits)
        // so you get a number 0..=31 * 256 (0..=7936)
        let offset = ((higher_ord_byte & 0x1F) as usize) * 256;

        // we add in the last byte, which is in range 0..256
        let byte = self.bytes.pop().ok_or_else(|| {
            Error::new(
                ErrorKind::UnexpectedEof,
                "trying to read past end of file when getting next block byte",
            )
        })?;

        Ok(offset + 1 + byte as usize)
    }

    /// Reads byte strings from `Block` into `Scratch` until a byte flag >= 32 is encountered
    fn read_write_block_until_offset_byte(&mut self) -> Result<Option<u8>> {
        loop {
            let readlen = match self.bytes.pop() {
                Some(byte) => byte,
                None => break Ok(None),
            };

            // breaking on readlen >= 32
            if readlen >= 0x20 {
                break Ok(Some(readlen));
            }

            self.read_block_into_scratch(readlen + 1)?;
        }
    }

    fn read_block_into_scratch(&mut self, len: u8) -> Result<()> {
        for _ in 0..len {
            let byte = self.bytes.pop().ok_or_else(|| {
                Error::new(
                    ErrorKind::UnexpectedEof,
                    "unexpected end of file when reading next block byte",
                )
            })?;

            self.output.push(byte);
        }

        Ok(())
    }
}

/// Validates the file format and returns the initial offset to use.
fn parse_header(file: &mut Cursor<&[u8]>) -> Result<u32> {
    file.seek(Current(4))?; // skip 4 bytes

    let buf = &mut [0u8; 4];
    file.read_exact(buf)?;

    if buf != b"vulZ" {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "expecting 'vulZ' tag in position 0x04",
        ));
    }

    file.seek(Current(4))?; // skip 4 bytes
    file.read_exact(buf)?;
    if buf != &[4u8, 0, 0, 0] {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "expecting 0x04000000 at position 0x0c",
        ));
    }

    file.seek(Current(8))?; // skip 8 bytes

    // read 4 bytes into the offset as a 32 bit little endian
    file.read_u32::<LittleEndian>()
}

/// Given the initial offset, follows the location pointers that sit in blocks of 2048 bytes.
/// After following 3 jumps, reads in the value, appending it to the location of blocks.
///
/// **Note: The block locations is in read _order_. It should be reversed when being consumed.
fn populate_block_locations(file: &mut Cursor<&[u8]>, start_offset: u64) -> Result<BlockLocations> {
    let mut locs = BlockLocations::new();

    // we read in 8 bytes at a time, so that is 256 * 8 = 2048

    for pos in (0..256).map(|j| start_offset + j * 8) {
        file.seek(Start(pos))?;
        let jumpto = file.read_u64::<LittleEndian>()?;

        if jumpto == 0 {
            continue;
        }

        // first jumps
        for pos in (0..256).map(|j| jumpto + j * 8) {
            file.seek(Start(pos))?;
            let jumpto = file.read_u64::<LittleEndian>()?;

            if jumpto == 0 {
                continue;
            }

            // second jumps
            for pos in (0..256).map(|j| jumpto + j * 8) {
                file.seek(Start(pos))?;
                let jumpto = file.read_u64::<LittleEndian>()?;

                if jumpto == 0 {
                    continue;
                }

                // third jumps -- seek first then use cursor propagation
                file.seek(Start(jumpto))?;

                // reads 8 bytes at a time -- 2048 total
                for _ in 0..256 {
                    let loc = file.read_u64::<LittleEndian>()?;

                    if loc > 0 {
                        locs.push(loc);
                    }
                }
            }
        }
    }

    Ok(locs)
}

fn extract_blocks(locations: BlockLocations, mut input: Cursor<&[u8]>) -> Result<Bytes> {
    let mut buffer = Vec::new();

    for loc in locations {
        input.seek(Start(loc))?;
        let block_len = input.read_u32::<LittleEndian>()? as usize;
        // this represents the trailing items (including `FA FB FC FD` pattern) but is
        let _block_len_with_trailing = input.read_i32::<LittleEndian>()?;

        let mut block = vec![0u8; block_len];
        input.read_exact(&mut block)?;

        buffer.extend_from_slice(&block);
    }

    Ok(buffer)
}

/// Starting at the buffer length minus `offset`, each byte is read and then _pushed_
/// onto the end of the buffer, for `len`.
/// **Note: This operates on a per-byte basis.**
///
/// If `offset` is 0, a `0x00` byte is written (for `len`).
fn overwrite_within(buf: &mut Bytes, offset: usize, len: usize) -> Result<()> {
    let start = buf.len().checked_sub(offset).ok_or_else(|| {
        Error::new(
            ErrorKind::Other,
            "compression offset goes beyond scratch buffer length",
        )
    })?;
    let end = start + len + 2;

    for idx in start..end {
        let b = buf.get(idx).copied().unwrap_or(0);
        buf.push(b);
    }

    Ok(())
}

#[cfg(test)]
mod conversion_tests {
    #[test]
    fn bulk_test_conversion() {
        let names = [
            "tri_1_blue",
            "tri_1_red",
            "tri_1_tri_3_sep_red",
            "tri_2_blue",
            "tri_2_red",
            "tri_123_sep_red",
            "sda",
        ];

        for name in names {
            convert_test(name);
        }
    }

    #[test]
    fn bulk_test_conversion_large_tris() {
        let names = ["WD9_REHAB_SLOPE _ EXTENDED", "dfg"];

        for name in names {
            convert_test(name);
        }
    }

    fn convert_test(name: &str) {
        println!("TESTING: '{}'", name);
        let dir = "test/vulcan-tries";
        let i = &std::fs::read(format!("{}/{}.00t", dir, name)).unwrap();
        let exp = std::fs::read(format!("{}/{}-converted.00t", dir, name)).unwrap();

        let unpacked = super::unpack_vulcan_8_00t(i).expect("failed conversion");

        assert_eq!(exp, unpacked, "unpacked does not match converted");
    }
}
