#!/usr/bin/env python3
"""
diffusion_score.py

Read a stereo WAV file and render a multi-page “diffusion score” PDF:
- Each system (line) has an upper area with waveform
- Lower area is blank “notes” space (optionally with 5 staff lines)
- Audio is split across multiple lines and pages

Dependencies:
  pip install numpy matplotlib soundfile
(or replace soundfile with scipy.io.wavfile if you prefer)

Usage examples:
  python diffusion_score.py input.wav
  python diffusion_score.py input.wav -o score.pdf --sec-per-line 12 --lines-per-page 6
  python diffusion_score.py input.wav --mono --no-staff
"""

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyBboxPatch



mpl.rcParams.update({
    "text.usetex": False,          # python-only text rendering
    "font.family": "monospace",    # request monospace
    "font.monospace": [
        "DejaVu Sans Mono",        # ships with matplotlib
        "Liberation Mono",
        "Nimbus Mono L",
        "Courier New",
        "Courier",
        "monospace",
    ],
    "mathtext.fontset": "dejavusans",  # math in a compatible style
})

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit("Missing dependency: soundfile. Install with: pip install soundfile") from e


def read_audio(path: str):
    data, sr = sf.read(path, always_2d=True)  # shape (N, C)
    if data.shape[1] < 2:
        # still accept mono, but treat as 1-channel
        pass



    return data.astype(np.float32), sr


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 1:
        return data[:, 0]
    return np.mean(data[:, :2], axis=1)


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    m = np.max(np.abs(x))
    if m < eps:
        return x
    return x / 1


def downsample_for_plot(x: np.ndarray, max_points: int) -> np.ndarray:
    """
    Simple decimation for plotting speed.
    Keeps shape; if x is (N,) returns (M,), if (N,2) returns (M,2).
    """
    n = x.shape[0]
    if n <= max_points:
        return x
    step = int(math.ceil(n / max_points))
    return x[::step]


def draw_staff(ax, x0, y0, w, h, n_lines=5, lw=0.6, alpha=0.8):
    """
    Draw n_lines staff lines centered vertically in the given rectangle (in ax coords).
    """
    if n_lines <= 0:
        return
    # Keep staff in the middle band of the notes area
    pad_y = 0.18 * h
    usable_h = h - 2 * pad_y
    if usable_h <= 0:
        return
    ys = np.linspace(y0 + pad_y, y0 + pad_y + usable_h, n_lines)
    for y in ys:
        ax.plot([x0, x0 + w], [y, y], color="black", lw=lw, alpha=alpha, solid_capstyle="butt")


def render_pdf(
    wav_path: str,
    out_pdf: str,
    sec_per_line: float = 10.0,
    lines_per_page: int = 6,
    dpi: int = 300,
    page_size: str = "A4",
    waveform_mode: str = "stereo",  # "stereo" or "mono"
    show_staff: bool = True,
    staff_lines: int = 5,
    max_plot_points: int = 2500,
):
    data, sr = read_audio(wav_path)

    # Use only first two channels if more are present
    if data.shape[1] >= 2:
        stereo = data[:, :2]
    else:
        stereo = data[:, :1]

    for i in range(data.shape[1]):
        print(i)
        data[:,i]=data[:,i]/max(data[:,i])

    total_samples = stereo.shape[0]
    total_sec = total_samples / sr

    samples_per_line = int(round(sec_per_line * sr))
    if samples_per_line <= 0:
        raise ValueError("sec_per_line must be > 0")

    n_lines_total = int(math.ceil(total_samples / samples_per_line))
    n_pages = int(math.ceil(n_lines_total / lines_per_page))

    # Page sizes in inches (matplotlib expects inches)
    # A4: 8.27 x 11.69, Letter: 8.5 x 11
    if page_size.lower() == "letter":
        fig_w, fig_h = 8.5, 11.0
    else:
        fig_w, fig_h = 8.27, 11.69

    wav_name = Path(wav_path).name

    with PdfPages(out_pdf) as pdf:
        for page_idx in range(n_pages):
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            fig.patch.set_facecolor("white")

            # A single full-page axes used as a drawing canvas in normalized coordinates
            canvas = fig.add_axes([0, 0, 1, 1])
            canvas.set_axis_off()
            canvas.set_xlim(0, 1)
            canvas.set_ylim(0, 1)

            # Layout (all in normalized figure coords)
            left = 0.06
            right = 0.96
            top = 0.95
            bottom = 0.05
            usable_w = right - left
            usable_h = top - bottom

            header_h = 0.06
            footer_h = 0.03
            systems_h = usable_h - header_h - footer_h

            # Header
            canvas.text(left, top+0.025, f"Diffusion Score: {wav_name}", ha="left", va="top", fontsize=12)
            canvas.text(
                right,
                top+0.025,
                f"Page {page_idx + 1}/{n_pages}",
                ha="right",
                va="top",
                fontsize=10,
            )

            # System geometry
            gap = 0.066
            system_h = (systems_h - gap * (lines_per_page - 1)) / lines_per_page

            # Each system: upper waveform box + lower notes box
            # Proportion similar to your sketch
            upper_frac = 0.45
            lower_frac = 0.55

            for line_on_page in range(lines_per_page):
                global_line = page_idx * lines_per_page + line_on_page
                if global_line >= n_lines_total:
                    break

                # System vertical placement (top-down)
                y_top = top - header_h - (line_on_page * (system_h + gap))
                y0 = y_top - system_h

                # Boxes
                upper_h = system_h * upper_frac
                lower_h = system_h * lower_frac


                canvas.add_patch(
                FancyBboxPatch(
                    (left, y0+0.025),
                    usable_w,
                    upper_h+lower_h,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    fill=False,
                    linewidth=1,
                    edgecolor="black"
                    )
                    )
               # Lower box rectangle (rounded)
                # canvas.add_patch(
                #     FancyBboxPatch(
                #         (left, y0),
                #         usable_w,
                #         lower_h,
                #         boxstyle="round,pad=0.01,rounding_size=0.02",
                #         fill=False,
                #         linewidth=2,
                #         edgecolor="black",
                #     )
                # )



                # Notes area staff lines (optional)
                if show_staff:
                    draw_staff(
                        canvas,
                        x0=left + 0.01,
                        y0=y0 + 0.02 * lower_h,
                        w=usable_w - 0.02,
                        h=lower_h - 0.04 * lower_h,
                        n_lines=staff_lines,
                        lw=0.6,
                        alpha=0.9,
                    )

                # Segment bounds
                s0 = global_line * samples_per_line
                s1 = min((global_line + 1) * samples_per_line, total_samples)
                seg = stereo[s0:s1]

                # Build waveform trace(s)
                if waveform_mode == "mono":
                    wv = normalize(to_mono(seg))
                    wv = downsample_for_plot(wv, max_plot_points)
                    t = np.linspace(0, (s1 - s0) / sr, num=wv.shape[0], endpoint=False)

                    # Create an axes inside the upper box
                    ax = fig.add_axes([left, y0 + lower_h +0.02, usable_w, upper_h])
                    ax.plot(t, wv, lw=0.8, color="black")
                    ax.patch.set_facecolor('none')
                else:
                    # stereo: plot L and R as two traces
                    if seg.shape[1] == 1:
                        L = normalize(seg[:, 0])
                        R = L
                    else:
                        L = normalize(seg[:, 0])
                        R = normalize(seg[:, 1])

                    stacked = np.column_stack([L, R])
                    stacked = downsample_for_plot(stacked, max_plot_points)
                    Lp, Rp = stacked[:, 0], stacked[:, 1]
                    t = np.linspace(0, (s1 - s0) / sr, num=stacked.shape[0], endpoint=False)

                    ax = fig.add_axes([left, y0 + lower_h+0.02, usable_w, upper_h])
                    ax.plot(t, Lp, lw=0.7, color="black")
                    ax.plot(t, Rp, lw=0.7, color="gray", alpha=0.75)
                    ax.patch.set_facecolor('none')

                # Style waveform axes to look like a “staff”
                ax.set_xlim(0, (s1 - s0) / sr if (s1 - s0) > 0 else 1.0)
                ax.set_ylim(-1.05, 1.05)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)


                # Time arrow underneath the lower box (like in sketch)
                arrow_y = y0 + 0.1
                canvas.annotate(
                    "",
                    xy=(right, arrow_y),
                    xytext=(left, arrow_y),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
                )
                #canvas.text((left + right) / 2, arrow_y - 0.012, "time", ha="center", va="top", fontsize=9)



                # Timecode labels (absolute time in file)
                start_t = s0 / sr
                end_t = s1 / sr

                start_m = math.floor(start_t/60)
                start_s = int(start_t%60)

                end_m = math.floor(end_t/60)
                end_s = int(end_t%60)

                canvas.text(
                    left,
                    y0 + lower_h + upper_h + 0.04,
                    f"{start_m:0{2}d}" ":" f"{start_s:0{2}d}",
                    ha="left",
                    va="bottom",
                    fontsize=8,
                )
                # canvas.text(
                #     right,
                #     y0 + lower_h + upper_h + 0.04,
                #     f"{end_m:0{2}d}" ":"  f"{end_s:0{2}d}",
                #     ha="right",
                #     va="bottom",
                #     fontsize=8,
                # )

            # Footer
            canvas.text(
                left,
                bottom + 0.002,
                f"Duration: {math.floor(total_sec/60):0{2}d}:{int(total_sec%60) :d} — {sr} Hz — {('stereo' if stereo.shape[1] >= 2 else 'mono')}",
                ha="left",
                va="bottom",
                fontsize=8,
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Create a diffusion-score PDF from a WAV file.")
    ap.add_argument("wav", help="Input WAV file (stereo preferred)")
    ap.add_argument("-o", "--out", default=None, help="Output PDF (default: <input>.pdf)")
    ap.add_argument("--sec-per-line", type=float, default=10.0, help="Seconds per line/system (default: 10)")
    ap.add_argument("--lines-per-page", type=int, default=6, help="Systems per page (default: 6)")
    ap.add_argument("--page-size", choices=["A4", "Letter"], default="A4", help="Page size (default: A4)")
    ap.add_argument("--dpi", type=int, default=300, help="Render DPI (default: 300)")
    ap.add_argument("--mono", action="store_true", help="Plot mono waveform instead of stereo traces")
    ap.add_argument("--no-staff", action="store_true", help="Do not draw staff lines in notes area")
    ap.add_argument("--staff-lines", type=int, default=5, help="Number of staff lines (default: 5)")
    ap.add_argument("--max-plot-points", type=int, default=2500, help="Max plotted points per line (default: 2500)")
    args = ap.parse_args()

    out_pdf = args.out
    if out_pdf is None:
        out_pdf = str(Path(args.wav).with_suffix(".pdf"))

    render_pdf(
        wav_path=args.wav,
        out_pdf=out_pdf,
        sec_per_line=args.sec_per_line,
        lines_per_page=args.lines_per_page,
        dpi=args.dpi,
        page_size=args.page_size,
        waveform_mode="mono" if args.mono else "stereo",
        show_staff=(not args.no_staff),
        staff_lines=args.staff_lines,
        max_plot_points=args.max_plot_points,
    )
    print(f"Wrote: {out_pdf}")


if __name__ == "__main__":
    main()
