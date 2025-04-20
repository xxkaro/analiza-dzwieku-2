import numpy as np
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import librosa


def frequency_spectrum(y, sr):
    N = len(y)
    y_fft = fft(y)
    freqs = fftfreq(N, 1/sr)
    
    idx = np.arange(N//2)
    freqs_list = freqs[idx].tolist()
    amplitude_list = np.abs(y_fft[idx]).tolist()
    return freqs_list, amplitude_list

def apply_window(y, sr, window_type='hamming', frame_size=1024):
    if window_type == 'rectangular':
        window = np.ones(frame_size)
    elif window_type == 'triangular':
        window = np.bartlett(frame_size)
    elif window_type == 'hamming':
        window = np.hamming(frame_size)
    elif window_type == 'hann':
        window = np.hanning(frame_size)
    else:
        raise ValueError("Unknown window type")

    frame = y[:frame_size] * window
    t_frame = np.arange(frame_size) / sr
    t_list = t_frame.tolist()
    frame_list = frame.tolist()

    freqs_list, amplitude_list = frequency_spectrum(frame, sr)

    return t_list, frame_list, freqs_list, amplitude_list

def generate_test_y(frequency=1000, sr=10000, duration=1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * frequency * t)
    return t, y

def compute_volume(spectrum):
    return np.mean(np.square(np.abs(spectrum)))

def compute_frequency_centroid(frequencies, spectrum):
    magnitude = np.abs(spectrum)
    return np.sum(frequencies * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0

def compute_bandwidth(frequencies, spectrum):
    fc = compute_frequency_centroid(frequencies, spectrum)
    power = np.square(np.abs(spectrum))
    return np.sqrt(np.sum(((frequencies - fc) ** 2) * power) / np.sum(power)) if np.sum(power) > 0 else 0

def compute_band_energy(frequencies, spectrum, f0, f1):
    band_mask = (frequencies >= f0) & (frequencies < f1)
    return np.sum(np.square(np.abs(spectrum[band_mask])))

def compute_band_energy_ratios(frequencies, spectrum):
    total_energy = np.sum(np.square(np.abs(spectrum)))
    subbands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]
    ratios = []
    for f0, f1 in subbands[:-1]:  # tylko 3 pierwsze, jak w opisie
        band_energy = compute_band_energy(frequencies, spectrum, f0, f1)
        ratio = band_energy / total_energy if total_energy > 0 else 0
        ratios.append(ratio)
    return ratios  # [ERSB1, ERSB2, ERSB3]

def get_band_indices(frequencies, f_low=100, f_high=6000):
    f_low_idx = np.argmin(np.abs(frequencies - f_low))
    f_high_idx = np.argmin(np.abs(frequencies - f_high))
    return f_low_idx, f_high_idx

def compute_spectral_flatness(spectrum, frequencies, f_low=100, f_high=6000):
    mask = (frequencies >= f_low) & (frequencies <= f_high)
    band = spectrum[mask]

    power = np.square(np.abs(band))
    geometric_mean = np.exp(np.mean(np.log(power + 1e-12))) 
    arithmetic_mean = np.mean(power)

    return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 1

def compute_spectral_crest(spectrum, frequencies, f_low=100, f_high=6000):
    mask = (frequencies >= f_low) & (frequencies <= f_high)
    band = spectrum[mask]

    power = np.square(np.abs(band))
    return np.max(power) / np.mean(power) if np.mean(power) > 0 else 0


def extract_frame_features(y, sr, frame_size_ms, hop_length):
    frame_size = int(sr * frame_size_ms / 1000)

    frame_features = {
        'volume': [],
        'centroid': [],
        'bandwidth': [],
        'energy_ratios': [],
        'flatness': [],
        'crest': []      
    }

    for start in range(0, len(y) - frame_size + 1, hop_length):
        frame = y[start:start + frame_size]
        frequencies = get_frequencies(len(frame), sr)
        spectrum = fft(frame)[:frame_size // 2]

        volume = compute_volume(spectrum)
        centroid = compute_frequency_centroid(frequencies, spectrum)
        bandwidth = compute_bandwidth(frequencies, spectrum)
        energy_ratios = compute_band_energy_ratios(frequencies, spectrum)
        flatness = compute_spectral_flatness(spectrum, frequencies)
        crest = compute_spectral_crest(spectrum, frequencies)

        frame_features['volume'].append(volume)
        frame_features['centroid'].append(centroid)
        frame_features['bandwidth'].append(bandwidth)
        frame_features['energy_ratios'].append(energy_ratios)
        frame_features['flatness'].append(flatness)
        frame_features['crest'].append(crest)

    return frame_features

def get_frequencies(N, sr):
    return np.fft.fftfreq(N, d=1/sr)[:N // 2]


def plot_feature(time_axis, feature_values, feature_name, mode):
    """
    Tworzy interaktywny wykres dla wartości cechy w czasie.
    """

    frame_numbers = np.arange(1, len(feature_values) + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_axis, 
        y=feature_values, 
        mode='lines', 
        name=feature_name,
        hovertemplate=
            f"<b>Numer {mode}:</b> %{{customdata[0]}}<br>" +
            "<b>Czas:</b> %{x:.3f} s" + "<br>" +
            f"<b>{feature_name}:</b> %{{y:.3f}}" + "<extra></extra>", 
        customdata=np.array([frame_numbers]).T  
    ))

    fig.update_layout(
        title=f"{feature_name}",
        title_x=0.5,
        xaxis_title="Czas (s)",
        yaxis_title=f"{feature_name}",
        hovermode="closest" 
    )

    return fig


def plot_ersb(time_axis, energy_ratios_list, mode):

    ersb1 = [e[0] for e in energy_ratios_list]
    ersb2 = [e[1] for e in energy_ratios_list]
    ersb3 = [e[2] for e in energy_ratios_list]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis,
                            y=ersb1,
                            mode='lines+markers',
                            name='ERSB1 (0–630 Hz)',
                            hovertemplate=
                            f"<b>Numer {mode}:</b> %{{customdata[0]}}<br>" +
                            "<b>Czas:</b> %{x:.3f} s" + "<br>" +
                            f"<b>ERSB1 (0–630 Hz)'</b> %{{y:.3f}}" + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=time_axis,
                            y=ersb2,
                            mode='lines+markers',
                            name='ERSB2 (630–1720 Hz)',
                            hovertemplate=
                            f"<b>Numer {mode}:</b> %{{customdata[0]}}<br>" +
                            "<b>Czas:</b> %{x:.3f} s" + "<br>" +
                            f"<b>ERSB2 (630–1720 Hz)'</b> %{{y:.3f}}" + "<extra></extra>"))
    fig.add_trace(go.Scatter(x=time_axis,
                            y=ersb3,
                            mode='lines+markers',
                            name='ERSB3 (1720–4400 Hz)',
                            hovertemplate=
                            f"<b>Numer {mode}:</b> %{{customdata[0]}}<br>" +
                            "<b>Czas:</b> %{x:.3f} s" + "<br>" +
                            f"<b>ERSB3 (1720–4400 Hz)'</b> %{{y:.3f}}" + "<extra></extra>"))

    fig.update_layout(
        title='ERSB (Energy Ratio Subbands)',
        xaxis_title='Numer ramki',
        yaxis_title='Proporcja energii',
        yaxis=dict(range=[0, 1]),
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        height=400
    )

    return fig


