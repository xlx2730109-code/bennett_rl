# 在 SolidWorks 打开零件或装配体。
# 打开“评估 → 质量属性”。
# 指定与你 URDF link 坐标系一致的“输出坐标系”。
# 点击“复制到剪贴板”。
# 把完整的中文质量属性文本粘贴到终端。
# 最后连续按两次回车，脚本开始转换。



import re
import sys


def parse_solidworks_to_urdf(sw_text):
    """解析SolidWorks质量属性文本，输出URDF inertial XML"""

    # 提取质量（支持"质量 = xxx 千克"和"质量（用户覆盖） = xxx 千克"两种格式）
    mass_match = re.search(r'质量(?:（[^）]*）)?\s*=\s*([\d.eE+-]+)\s*千克', sw_text)
    if not mass_match:
        raise ValueError("未找到质量数据（格式：质量 = xxx 千克 或 质量（用户覆盖） = xxx 千克）")
    mass = float(mass_match.group(1))

    # 提取质心
    com_pattern = r'质心.*?X\s*=\s*([-\d.eE+]+).*?Y\s*=\s*([-\d.eE+]+).*?Z\s*=\s*([-\d.eE+]+)'
    com_match = re.search(com_pattern, sw_text, re.DOTALL)
    if not com_match:
        raise ValueError("未找到质心数据")
    cx = float(com_match.group(1))
    cy = float(com_match.group(2))
    cz = float(com_match.group(3))

    # 检测质心单位，如果是毫米则转换为米
    unit_match = re.search(r'质心\s*[:：]\s*\(\s*(.*?)\s*\)', sw_text)
    if unit_match:
        unit = unit_match.group(1).strip()
        if '毫米' in unit:
            cx /= 1000.0
            cy /= 1000.0
            cz /= 1000.0
            print(f"[提示] 检测到质心单位为毫米，已自动转换为米")

    # 提取"由重心决定，并且对齐输出的坐标系"的惯性张量
    inertia_pattern = (
        r'由重心决定，并且对齐输出的坐标系.*?'
        r'Lxx\s*=\s*([-\d.eE+]+)\s*Lxy\s*=\s*([-\d.eE+]+)\s*Lxz\s*=\s*([-\d.eE+]+)\s*'
        r'Lyx\s*=\s*([-\d.eE+]+)\s*Lyy\s*=\s*([-\d.eE+]+)\s*Lyz\s*=\s*([-\d.eE+]+)\s*'
        r'Lzx\s*=\s*([-\d.eE+]+)\s*Lzy\s*=\s*([-\d.eE+]+)\s*Lzz\s*=\s*([-\d.eE+]+)'
    )
    inertia_match = re.search(inertia_pattern, sw_text, re.DOTALL)
    if not inertia_match:
        raise ValueError("未找到惯性张量数据（由重心决定，并且对齐输出的坐标系）")

    ixx = float(inertia_match.group(1))
    ixy = float(inertia_match.group(2))
    ixz = float(inertia_match.group(3))
    iyy = float(inertia_match.group(5))
    iyz = float(inertia_match.group(6))
    izz = float(inertia_match.group(9))

    # 检测惯性张量单位，如果是 克*平方毫米 则转换为 千克*平方米
    inertia_unit_match = re.search(r'惯性张量\s*[:：]\s*\(\s*(.*?)\s*\)', sw_text)
    if inertia_unit_match:
        inertia_unit = inertia_unit_match.group(1).strip()
        if '克' in inertia_unit and '千克' not in inertia_unit:
            # 克*平方毫米 -> 千克*平方米: ×1e-9
            factor = 1e-9
            ixx *= factor
            ixy *= factor
            ixz *= factor
            iyy *= factor
            iyz *= factor
            izz *= factor
            print(f"[提示] 检测到惯性张量单位为 克*平方毫米，已自动转换为 千克*平方米")

    # 生成URDF格式
    urdf = f"""    <inertial>
      <origin
        xyz="{cx} {cy} {cz}"
        rpy="0 0 0" />
      <mass
        value="{mass}" />
      <inertia
        ixx="{ixx}"
        ixy="{ixy}"
        ixz="{ixz}"
        iyy="{iyy}"
        iyz="{iyz}"
        izz="{izz}" />
    </inertial>"""

    return urdf


def main():
    print("=" * 50)
    print("SolidWorks 质量属性 -> URDF Inertial 转换工具")
    print("=" * 50)
    print("请粘贴SolidWorks质量属性文本（连续两个空行结束输入）：")
    print("-" * 50)

    lines = []
    empty_count = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append(line)
        else:
            empty_count = 0
            lines.append(line)

    sw_text = "\n".join(lines)

    if not sw_text.strip():
        print("错误：未输入任何数据", file=sys.stderr)
        sys.exit(1)

    try:
        # 以"质量 = "为标记切分多个块，避免块内空行干扰
        parts = re.split(r'\n(?=质量\s*=\s*[\d.eE+-])', sw_text.strip())
        blocks = [p.strip() for p in parts if p.strip()]

        if not blocks:
            print("错误：未找到有效数据块", file=sys.stderr)
            sys.exit(1)

        print("\n" + "=" * 50)
        print("URDF Inertial 输出：")
        print("=" * 50)

        for i, block in enumerate(blocks):
            result = parse_solidworks_to_urdf(block)
            if i > 0:
                print()  # 块间空行分隔
            print(result)
    except ValueError as e:
        print(f"解析错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
