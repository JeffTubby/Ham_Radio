import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplcursors

DATA_PATH = pathlib.Path(__file__).parent / 'data' / 'VNA_010926.zplot.csv'
SYSTEM_IMPEDANCE = 50.0

df = pd.read_csv(DATA_PATH, encoding='latin1')

frequency = df['Frequency(Hz)'].to_numpy()
return_loss = df['Return Loss(dB)'].to_numpy()
phase = np.deg2rad(df['±Phase(deg)'].to_numpy())

# Convert the measured reflection coefficient to impedance.
reflection_coefficient = 10 ** (-return_loss / 20) * np.exp(1j * phase)
impedance = SYSTEM_IMPEDANCE * (1 + reflection_coefficient) / (1 - reflection_coefficient)

plt.style.use('ggplot')
fig, axis = plt.subplots(figsize=(6,6))
axis.set_facecolor('#f8fafc')
scatter = axis.scatter(
    impedance.real,
    impedance.imag,
    c=frequency / 1e6,
    cmap='viridis',
    s=18,
)
nyquist_line, = axis.plot(
    impedance.real,
    impedance.imag,
    color='#94a3b8',
    linewidth=0.8,
    zorder=1,
)
colorbar = fig.colorbar(scatter, ax=axis)
colorbar.set_label('Frequency (MHz)')

axis.axhline(0, color='#475569', linewidth=0.8)
axis.axvline(0, color='#475569', linewidth=0.8)
axis.set_title('Nyquist Plot: Impedance')
axis.set_xlabel('Real(Z) (ohms)')
axis.set_ylabel('Imag(Z) (ohms)')
axis.grid(True, which='major', color='#cbd5e1', linewidth=0.8, alpha=0.8)
axis.grid(True, which='minor', color='#e2e8f0', linewidth=0.5, alpha=0.7)
axis.minorticks_on()
axis.set_aspect('equal', adjustable='datalim')

cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect('add')
def show_frequency_and_impedance(selection):
    data_index = int(selection.index)
    row = df.iloc[data_index]
    selected_impedance = impedance[data_index]
    selection.annotation.set_text(
        f"Frequency: {row['Frequency(Hz)'] / 1e6:.3f} MHz\n"
        f"Z: {selected_impedance.real:.2f} + {selected_impedance.imag:.2f}j ohms\n"
        f"Return loss: {row['Return Loss(dB)']:.1f} dB\n"
        f"Phase: {row['±Phase(deg)']:.1f} deg"
    )
    selection.annotation.get_bbox_patch().set(
        facecolor='white',
        edgecolor='#334155',
        alpha=0.95,
    )

fig.tight_layout()
plt.show()
