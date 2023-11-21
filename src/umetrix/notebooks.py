import base64
import io
import itertools
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors, colormaps

from umetrix.core import METRICS


DARK_CONTEXT = {
    "axes.edgecolor": "gray",
    "xtick.color": "gray",
    "ytick.color": "gray",
    "axes.labelcolor": "gray",
    "font.size": 18,
}


def _text_color_based_on_value(value: float, cmap: colors.Colormap) -> str:
    rgb = cmap(value)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "k" if luminance > 0.5 else "w"


def _header() -> str:
    css = (
        ".row {display: inline-flex; align-items: flex-start;}\n"
        ".metrics {float: left;}\n"
        ".confusion {width: 200px; float: left;}"
    )
    return f"<html><style>{css}</style><body>"


def _footer() -> str:
    return "</body></html>"


def render_metrics_html(metrics) -> str:
    """Render the metrics to HTML"""

    html_table = _render_table(metrics)
    encoded_cm = _render_confusion(metrics)
    html_strict = (
        f"<p>Strict matching (IoU threshold: {metrics.iou_threshold})</p>"
        if metrics.strict
        else ""
    )

    html = (
        _header()
        + "<p><h3>Segmentation Metrics</h3></p>"
        + html_strict
        + "<div class='row'><div class='metrics'>"
        + html_table
        + "</div><div class='confusion'>"
        + f"<img src='data:image/png;base64,{encoded_cm}' width=200px/>"
        + "</div></div>"
        + _footer()
    )

    return html


def _render_table(metrics) -> str:
    """Render the table of results"""

    def _get_f_string(m):
        val = getattr(metrics, m)
        return f"{val:.3f}" if isinstance(val, float) else f"{val:d}"

    return (
        "<table><tr><th>Metric</th><th><Value></th></tr>"
        + "".join(
            [f"<tr><td>{m}</td><td>" + _get_f_string(m) + "</td></tr>" for m in METRICS]
        )
        + "</table>"
    )


def _render_confusion(metrics, *, cmap: str = "Blues") -> str:
    """Render a confusion matrix as an image"""
    grid = np.zeros((2, 2), dtype=float)
    grid[1, 1] = metrics.n_true_positives
    grid[0, 1] = metrics.n_false_positives
    grid[1, 0] = metrics.n_false_negatives
    cmap = colormaps[cmap]

    with plt.rc_context(DARK_CONTEXT):
        _, ax = plt.subplots(figsize=(4, 4))
        ax.pcolor(grid, cmap=cmap)
        ax.set_xticks([0.5, 1.5], labels=["Negative", "Positive"])
        ax.set_yticks(
            [0.5, 1.5], labels=["Negative", "Positive"], rotation=90, va="center"
        )

        for i, j in itertools.product(range(2), range(2)):
            ax.text(
                i + 0.5,
                j + 0.5,
                grid[i, j].astype(int),
                ha="center",
                va="center",
                color=_text_color_based_on_value(grid[i, j] / np.max(grid), cmap),
            )
        ax.set_ylabel("Predicted")
        ax.set_xlabel("Ground truth")
    stream = io.BytesIO()
    plt.savefig(stream, format="png", bbox_inches="tight", transparent=True)
    plt.close()
    return base64.b64encode(stream.getvalue()).decode("utf-8")
