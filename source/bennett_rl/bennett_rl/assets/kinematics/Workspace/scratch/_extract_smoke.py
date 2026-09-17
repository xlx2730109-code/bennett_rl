"""Scratch: extract a few frames of the smoke GIF for visual check."""
from PIL import Image

SRC = r"E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\assets\kinematics\_smoke_local.gif"
im = Image.open(SRC)
for i in (0, 6, 12, 18):
    im.seek(i) or im.convert("RGB").save(
        rf"E:\Project\Isaaclab\bennett_rl\outputs\_smk2\f{i:02d}.png")
print("ok", im.n_frames, "frames")
