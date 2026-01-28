# Diffusion Score

Read a wave file and create a diffusion score.
It shows the waveform with timestamps and space for annotation:

- **Upper staff**: waveform (mono or stereo)
- **Lower staff**: empty space for diffusion notes
- Audio is split into **lines (systems)** and **pages**
- Designed for printing and manual annotation during performance

The output is suitable for **acousmatic diffusion**, **live spatialization**, or **score-following** contexts.

---

## Features

- Reads mono or stereo WAV files
- Multi-page PDF output (A4 or Letter)
- Fixed-duration systems (e.g. 10 s per line)
- Rounded system frames (score-like appearance)
- Optional 5-line staff in notes area
- Stereo or mono waveform display
- Absolute time labels per system
- Print-ready layout (no UI elements)

---

## Dependencies

Install once:

    pip install numpy matplotlib soundfile



# Run it like this:

    python3 diff_score.py --sec-per-line 30 --lines-per-page 4 --staff-lines 0  --max-plot-points 5000 filename.wav


## Command Line Arguments

The script is controlled entirely via command line arguments to allow reproducible score layouts.

### Required

| Argument | Description |
|---------|------------|
| `wav` | Input WAV file (mono or stereo). The audio is never modified; it is only used for visualization. |

---

### Optional

#### Output and Page Layout

| Argument | Default | Description |
|---------|--------|------------|
| `-o`, `--out` | `<input>.pdf` | Output PDF file name. |
| `--page-size` | `A4` | Page format: `A4` or `Letter`. |
| `--dpi` | `300` | Rendering resolution for the PDF. Increase for high-quality printing. |

---

#### System (Line) Layout

| Argument | Default | Description |
|---------|--------|------------|
| `--sec-per-line` | `10.0` | Duration (in seconds) represented by one system (line). |
| `--lines-per-page` | `6` | Number of systems per page. Controls vertical density. |

---

#### Waveform Display

| Argument | Default | Description |
|---------|--------|------------|
| `--mono` | off | Plot a mono waveform (average of channels) instead of stereo traces. |
| `--max-plot-points` | `2500` | Maximum number of waveform samples plotted per system. Prevents slow rendering for long files. |

---

#### Notes Area (Lower Staff)

| Argument | Default | Description |
|---------|--------|------------|
| `--no-staff` | off | Disable staff lines in the notes area (leave it completely blank). |
| `--staff-lines` | `5` | Number of staff lines drawn in the notes area. Ignored if `--no-staff` is set. |


