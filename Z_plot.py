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
fig, axis = plt.subplots(figsize=(7,4))
axis.set_facecolor('#f8fafc')
real_line, = axis.plot(
    frequency / 1e6,
    impedance.real,
    label='Real impedance',
    color='#1769aa',
    linewidth=1.8,
)
imaginary_line, = axis.plot(
    frequency / 1e6,
    impedance.imag,
    label='Imaginary impedance',
    color='#d95f02',
    linewidth=1.8,
)
axis.set_title('Z-plot:Impedance vs Frequency')
axis.set_xlabel('Frequency (MHz)')
axis.set_ylabel('Impedance (ohms)')
axis.grid(True, which='major', color='#cbd5e1', linewidth=0.8, alpha=0.8)
axis.grid(True, which='minor', color='#e2e8f0', linewidth=0.5, alpha=0.7)
axis.minorticks_on()
axis.legend(loc='upper center', fontsize=10, frameon=True, fancybox=True, shadow=True)

cursor = mplcursors.cursor([real_line, imaginary_line], hover=True)

@cursor.connect("add")
def show_frequency_and_return_loss_and_phase(selection):
    data_index = int(selection.index)
    row = df.iloc[data_index]
    selected_impedance = impedance[data_index]
    selection.annotation.set_text(
        f"Frequency: {row['Frequency(Hz)'] / 1e6:.3f} MHz\n"
        f"{selection.artist.get_label()}: "
        f"{selected_impedance.real if selection.artist is real_line else selected_impedance.imag:.2f} ohms\n"
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

swr_path = pathlib.Path(__file__).parent / 'data' / 'VNA_SWR_010926.csv'
swr_df = pd.read_csv(swr_path, encoding='latin1')
swr_frequency = swr_df['Frequency(Hz)'].to_numpy()
swr = swr_df['SWR'].to_numpy()

swr_fig, swr_axis = plt.subplots(figsize=(7,4))
swr_axis.set_facecolor('#f8fafc')
swr_line, = swr_axis.plot(
    swr_frequency / 1e6,
    swr,
    label='SWR',
    color='#2a9d8f',
    linewidth=1.8,
)
swr_axis.set_title('SWR vs Frequency')
swr_axis.set_xlabel('Frequency (MHz)')
swr_axis.set_ylabel('SWR')
swr_axis.grid(True, which='major', color='#cbd5e1', linewidth=0.8, alpha=0.8)
swr_axis.grid(True, which='minor', color='#e2e8f0', linewidth=0.5, alpha=0.7)
swr_axis.minorticks_on()
swr_axis.legend(loc='upper center', fontsize=10, frameon=True, fancybox=True, shadow=True)

swr_cursor = mplcursors.cursor(swr_line, hover=True)

@swr_cursor.connect('add')
def show_frequency_and_swr(selection):
    data_index = int(selection.index)
    row = swr_df.iloc[data_index]
    selection.annotation.set_text(
        f"Frequency: {row['Frequency(Hz)'] / 1e6:.3f} MHz\n"
        f"SWR: {row['SWR']:.2f}\n"
        f"Return loss: {row['Return Loss(dB)']:.1f} dB"
    )
    selection.annotation.get_bbox_patch().set(
        facecolor='white',
        edgecolor='#334155',
        alpha=0.95,
    )

swr_fig.tight_layout()
plt.show()



