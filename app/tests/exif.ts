/** Shared by the measurement tests. */
/** EXIF orientation tag (0x0112) from a JPEG's APP1 segment; 1 if absent. */
export function exifOrientation(buf: Buffer): number {
  if (buf[0] !== 0xff || buf[1] !== 0xd8) return 1;
  let o = 2;
  while (o + 4 < buf.length) {
    if (buf[o] !== 0xff) return 1;
    const marker = buf[o + 1];
    const len = buf.readUInt16BE(o + 2);
    if (marker === 0xe1 && buf.toString("latin1", o + 4, o + 10) === "Exif\0\0") {
      const t = o + 10;
      const le = buf.toString("latin1", t, t + 2) === "II";
      const u16 = (p: number) => (le ? buf.readUInt16LE(p) : buf.readUInt16BE(p));
      const u32 = (p: number) => (le ? buf.readUInt32LE(p) : buf.readUInt32BE(p));
      const ifd = t + u32(t + 4);
      const n = u16(ifd);
      for (let i = 0; i < n; i++) {
        const e = ifd + 2 + i * 12;
        if (u16(e) === 0x0112) return u16(e + 8);
      }
      return 1;
    }
    if (marker === 0xda) return 1;
    o += 2 + len;
  }
  return 1;
}

