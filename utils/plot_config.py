import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import platform


def setup_chinese_font():
    sys_name = platform.system()
    if sys_name == "Darwin":
        candidates = ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"]
    elif sys_name == "Windows":
        candidates = ["Microsoft YaHei", "SimHei", "KaiTi"]
    else:
        candidates = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]

    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = None
    for c in candidates:
        if c in available:
            selected = c
            break
    if selected:
        plt.rcParams["font.sans-serif"] = [selected]
    plt.rcParams["axes.unicode_minus"] = False
