"""Scratch: extract frames of the FINAL GIF for a quality check."""
from PIL import Image

SRC = r"E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl\assets\kinematics\fig5_workspace_real_anim.gif"
OUT = r"E:\Project\Isaaclab\bennett_rl\outputs\_smk2"
im = Image.open(SRC)
for i in (6, 42):
    im.seek(i) or im.convert("RGB").save(rf"{OUT}\final_f{i:02d}.png")
print("ok", im.n_frames, "frames,", im.size)
