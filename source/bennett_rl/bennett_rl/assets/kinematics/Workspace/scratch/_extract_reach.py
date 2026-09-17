"""Scratch: extract check frames from the final reachable-surface GIFs."""
from PIL import Image

BASE = (r"E:\Project\Isaaclab\bennett_rl\source\bennett_rl\bennett_rl"
        r"\assets\kinematics\Workspace")
for tag, src in (("p2", rf"{BASE}\animation\fig5_workspace_real_anim.gif"),
                 ("foot", rf"{BASE}\animation\fig5_workspace_real_anim_foot.gif")):
    im = Image.open(src)
    print(tag, im.n_frames, "frames,", im.size)
    for i in (5, 30, 60, 90):
        im.seek(i)
        im.convert("RGB").save(rf"{BASE}\scratch\_fin_{tag}_{i:03d}.png")
print("ok")
