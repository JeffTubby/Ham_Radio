import pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplcursors

DATA_PATH = pathlib.Path(__file__).parent / 'data' / 'VNA_010926.zplot.csv'
SYSTEM_IMPEDANCE = 50.0


def reflection_coefficient_from_impedance(impedance: Iterable[complex] | complex, z0: float = SYSTEM_IMPEDANCE):
    """Convert impedance values to the reflection coefficient gamma."""
    z = np.asarray(impedance, dtype=np.complex128)
    return (z - z0) / (z + z0)


def impedance_from_reflection_coefficient(gamma: Iterable[complex] | complex, z0: float = SYSTEM_IMPEDANCE):
    """Convert the reflection coefficient gamma to impedance."""
    g = np.asarray(gamma, dtype=np.complex128)
    return z0 * (1 + g) / (1 - g)


def load_vna_data(data_path: pathlib.Path = DATA_PATH):
    """Load the VNA CSV and return the reflection coefficient and frequency data."""
    df = pd.read_csv(data_path, encoding='latin1')
    frequency_hz = df['Frequency(Hz)'].to_numpy(dtype=float)
    return_loss_db = df['Return Loss(dB)'].to_numpy(dtype=float)
    phase_deg = df['±Phase(deg)'].to_numpy(dtype=float)

    gamma = 10 ** (-return_loss_db / 20.0) * np.exp(1j * np.deg2rad(phase_deg))
    impedance = impedance_from_reflection_coefficient(gamma, z0=SYSTEM_IMPEDANCE)
    return df, frequency_hz, gamma, impedance


def draw_smith_chart(
    ax: plt.Axes | None = None,
    data_path: pathlib.Path = DATA_PATH,
    z0: float = SYSTEM_IMPEDANCE,
    show: bool = True,
    output_path: pathlib.Path | str | None = None,
):
    """Draw a Smith chart showing the VNA measurements and the standard constant-R/X grid."""
    df, frequency_hz, gamma, impedance = load_vna_data(data_path)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    theta = np.linspace(0, 2 * np.pi, 720)

    # Outer reference circle.
    ax.plot(np.cos(theta), np.sin(theta), color='black', linewidth=1.3, alpha=0.95)

    # Constant resistance circles.
    resistance_values = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
    for resistance in resistance_values:
        if np.isclose(resistance, 0.0):
            ax.plot(np.cos(theta), np.sin(theta), color='#7a7a7a', linewidth=1.0, linestyle='-')
            continue
        center_x = resistance / (resistance + 1.0)
        radius = 1.0 / (resistance + 1.0)
        x = center_x + radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax.plot(x, y, color='#8b8b8b', linewidth=0.8, alpha=0.9)

    # Constant reactance circles.
    reactance_values = [0.2, 0.5, 1.0, 2.0, 5.0, -0.2, -0.5, -1.0, -2.0, -5.0]
    for reactance in reactance_values:
        if np.isclose(reactance, 0.0):
            continue
        center_x = 1.0
        center_y = 1.0 / reactance
        radius = abs(1.0 / reactance)
        x = center_x + radius * np.cos(theta)
        y = center_y + radius * np.sin(theta)
        ax.plot(x, y, color='#9ab7d6', linewidth=0.8, alpha=0.9)

    # Measurement trace with frequency coloring.
    norm = plt.Normalize(frequency_hz.min() / 1e6, frequency_hz.max() / 1e6)
    scat = ax.scatter(
        gamma.real,
        gamma.imag,
        c=frequency_hz / 1e6,
        cmap='plasma',
        norm=norm,
        s=26,
        edgecolors='white',
        linewidths=0.35,
        zorder=4,
    )
    cursor = mplcursors.cursor(scat, hover=True)

    @cursor.connect('add')
    def show_point_details(selection):
        index = int(selection.index)
        freq_hz = frequency_hz[index]
        gamma_value = gamma[index]
        z_value = impedance[index]
        return_loss = 20 * np.log10(abs(gamma_value)) * -1.0
        phase_deg = np.rad2deg(np.angle(gamma_value))
        swr = (1 + abs(gamma_value)) / (1 - abs(gamma_value)) if abs(gamma_value) < 1 else float('inf')

        selection.annotation.set_text(
            f"Frequency: {freq_hz / 1e6:.3f} MHz\n"
            f"Return Loss: {return_loss:.2f} dB\n"
            f"Phase: {phase_deg:.2f}°\n"
            f"Γ: {gamma_value.real:.4f} {gamma_value.imag:+.4f}j\n"
            f"Z: {z_value.real:.2f} + {z_value.imag:+.2f}j Ω\n"
            f"SWR: {swr:.3f}"
        )
        selection.annotation.set_fontsize(9)
        selection.annotation.get_bbox_patch().set(
            facecolor='white',
            edgecolor='#334155',
            alpha=0.95,
        )

    ax.plot(gamma.real, gamma.imag, color='#0b3d91', linewidth=2.0, alpha=0.9, zorder=3)

    ax.set_title('RF Smith Chart', fontsize=15, weight='bold', pad=18)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_frame_on(False)

    # Classic RF labels.
    ax.text(0.0, 1.08, 'OPEN', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(0.0, -1.08, 'SHORT', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(1.08, 0.0, 'R=0', ha='left', va='center', fontsize=9)
    ax.text(0.5, 0.0, 'R=1', ha='center', va='center', fontsize=9)
    ax.text(0.33, 0.0, 'R=2', ha='center', va='center', fontsize=9)
    ax.text(0.22, 0.0, 'R=5', ha='center', va='center', fontsize=9)
    ax.text(0.96, 0.48, 'X=+j1', ha='center', va='center', fontsize=8)
    ax.text(0.96, -0.48, 'X=-j1', ha='center', va='center', fontsize=8)
    ax.text(0.95, 0.88, 'X=+j0.2', ha='center', va='center', fontsize=8)
    ax.text(0.95, -0.88, 'X=-j0.2', ha='center', va='center', fontsize=8)
    ax.text(0.0, 0.0, '50 Ω', ha='center', va='center', fontsize=10, fontweight='bold', color='#1f2937')

    cbar = fig.colorbar(scat, ax=ax, pad=0.04)
    cbar.set_label('Frequency (MHz)', rotation=270, labelpad=18)
    cbar.ax.tick_params(labelsize=8)

    if output_path is not None:
        output_file = pathlib.Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_file, dpi=220)


    if output_path is not None:
        output_file = pathlib.Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_file, dpi=200)

    if show:
        plt.tight_layout()
        plt.show()

    return ax, gamma, impedance, df


if __name__ == '__main__':
    draw_smith_chart(show=True, output_path=pathlib.Path(__file__).with_name('smith_chart_plot.png'))
