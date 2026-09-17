"""Scratch: extract frames of the post-reorg smoke GIF."""
from PIL import Image

SRC = (r"E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl"
       r"\assets\kinematics\Workspace\animation\_smoke3.gif")
OUT = (r"E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl"
       r"\assets\kinematics\Workspace\scratch")
im = Image.open(SRC)
for i in (6, 12):
    im.seek(i) or im.convert("RGB").save(rf"{OUT}\smoke3_f{i:02d}.png")
print("ok", im.n_frames, "frames,", im.size)
