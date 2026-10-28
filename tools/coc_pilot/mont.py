import sys, glob
from PIL import Image, ImageDraw
H = sys.argv[1]
for d in sys.argv[2:]:
    fs = sorted(glob.glob(f"{H}/frames/{d}/*.jpg")); sel = [0, 2, 4, 6, 8, 10, 12, 16]
    M = Image.new("RGB", (400 * 4, 225 * 2))
    for n, i in enumerate(sel):
        im = Image.open(fs[i]).resize((400, 225)); ImageDraw.Draw(im).text((8, 8), f"{d} t={0.5*(i-4):+.1f}s", fill=(255, 255, 0))
        M.paste(im, ((n % 4) * 400, (n // 4) * 225))
    M.save(f"{H}/m{d}.jpg", quality=80)
