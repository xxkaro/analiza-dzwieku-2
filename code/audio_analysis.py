import numpy as np
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import librosa


def frequency_spectrum(y, sr, window_type='rectangular'):
    N = len(y)
    y_fft = fft(y)
    freqs = fftfreq(N, 1/sr)
    
    idx = np.arange(N//2)
    freqs_list = freqs[idx].tolist()
    amplitude_list = np.abs(y_fft[idx]).tolist()
    return freqs_list, amplitude_list

def apply_window(frame, window_type):
    if window_type == 'rectangular':
        window = np.ones(len(frame)) 
    elif window_type == 'triangular':
        window = np.bartlett(len(frame))  
    elif window_type == 'hamming':
        window = np.hamming(len(frame))  
    elif window_type == 'hanning':
        window = np.hanning(len(frame))  
    elif window_type == 'blackman':
        window = np.blackman(len(frame)) 
    else:
        raise ValueError(f"Nieznany typ okna: {window_type}")
    
    return frame * window


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

def compute_band_energy_ratios(sr, frequencies, spectrum):
    total_energy = np.sum(np.square(np.abs(spectrum)))
    # subbands = [(0, 630), (630, 1720), (1720, 4400), (4400, 11025)]
    f_max = sr / 2
    subbands = [
        (0, int(0.057 * f_max)),
        (int(0.057 * f_max), int(0.156 * f_max)),
        (int(0.156 * f_max), int(0.399 * f_max)),
        (int(0.399 * f_max), int(f_max))
    ]
    ratios = []
    for f0, f1 in subbands: 
        band_energy = compute_band_energy(frequencies, spectrum, f0, f1)
        ratio = band_energy / total_energy if total_energy > 0 else 0
        ratios.append(ratio)
    return ratios

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

def compute_cepstrum(y, window='rectangular'):
    frame = apply_window(y, window)
    spectrum = np.fft.fft(frame)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real

    return cepstrum


def compute_spectrogram(y, frame_size, hop_length, window='rectangular'):
    n_frames = (len(y) - frame_size) // hop_length + 1
    spectrogram = np.zeros((n_frames, frame_size // 2 + 1)) 

    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + frame_size]
        frame = apply_window(frame, window)
        spectrum = np.abs(np.fft.rfft(frame))  
        spectrogram[i, :] = spectrum

    return spectrogram

def extract_frame_features(y, sr, frame_size, hop_length, window='rectangular'):
    frame_features = {
        'volume': [],
        'centroid': [],
        'bandwidth': [],
        'energy_ratios': [],
        'flatness': [],
        'crest': [],  
    }

    for start in range(0, len(y) - frame_size + 1, hop_length):
        frame = y[start:start + frame_size]
        frame = apply_window(frame, window)
    
        if start + frame_size > len(y):
            frame = y[start:]  

        frequencies = get_frequencies(len(frame), sr)
        spectrum = fft(frame)[:len(frame) // 2]

        volume = compute_volume(spectrum)
        centroid = compute_frequency_centroid(frequencies, spectrum)
        bandwidth = compute_bandwidth(frequencies, spectrum)
        energy_ratios = compute_band_energy_ratios(sr, frequencies, spectrum)
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


def plot_spectrogram(spectrogram, sr, frame_size, hop_length):
    spectrogram = 20 * np.log10(spectrogram + 1e-6)  
    time_axis = np.arange(spectrogram.shape[0]) * hop_length / sr
    freq_axis = np.fft.fftfreq(frame_size, 1 / sr)[:frame_size // 2]

    fig = go.Figure(data=go.Heatmap(
        z=spectrogram.T,  
        x=time_axis,
        y=freq_axis,
        # zmin=np.percentile(spectrogram, 0.5), 
        # zmax=np.percentile(spectrogram, 99.5),
        colorbar=dict(title="Amplituda (dB)"),
    ))

    fig.update_layout(
        title="Spektrogram",
        xaxis_title="Czas (s)",
        yaxis_title="Częstotliwość (Hz)",
        yaxis=dict(type='log', tickvals=[10, 100, 1000, 10000], ticktext=['10 Hz', '100 Hz', '1 kHz', '10 kHz']), 
        template='plotly_white',
        height=450
    )

    return fig

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
        hovermode="closest",
        height=450, 
    )

    return fig


def plot_ersb(time_axis, energy_ratios_list, mode, sr):
    f_max = sr / 2
    subbands = [
        (0, int(0.057 * f_max)),
        (int(0.057 * f_max), int(0.156 * f_max)),
        (int(0.156 * f_max), int(0.399 * f_max)),
        (int(0.399 * f_max), int(f_max))
    ]
    band_labels = [f"{low}–{high} Hz" for (low, high) in subbands]

    ersb1 = [e[0] for e in energy_ratios_list]
    ersb2 = [e[1] for e in energy_ratios_list]
    ersb3 = [e[2] for e in energy_ratios_list]
    ersb4 = [e[3] for e in energy_ratios_list]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=ersb1,
        mode='lines',
        name=f'ERSB1 ({band_labels[0]})',
        line=dict(width=0.5, color='rgba(160, 197, 127, 0.8)'),
        stackgroup='one',
        groupnorm='fraction',
        hovertemplate=f"<b>{mode}:</b> %{{x}} s<br>ERSB1: %{{y:.3f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=ersb2,
        mode='lines',
        name=f'ERSB2 ({band_labels[1]})',
        line=dict(width=0.5, color='rgba(236, 90, 83, 0.8)'),
        stackgroup='one',
        hovertemplate=f"<b>{mode}:</b> %{{x}} s<br>ERSB2: %{{y:.3f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=ersb3,
        mode='lines',
        name=f'ERSB3 ({band_labels[2]})',
        line=dict(width=0.5, color='rgba(245, 191, 79, 0.8)'),
        stackgroup='one',
        hovertemplate=f"<b>{mode}:</b> %{{x}} s<br>ERSB3: %{{y:.3f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=time_axis,
        y=ersb4,
        mode='lines',
        name=f'ERSB4 ({band_labels[3]})',
        line=dict(width=0.5, color='rgba(43, 102, 194, 0.8)'),
        stackgroup='one',
        hovertemplate=f"<b>{mode}:</b> %{{x}} s<br>ERSB4: %{{y:.3f}}<extra></extra>"
    ))

    fig.update_layout(
        title='ERSB (Energy Ratio Subbands)',
        xaxis_title="Czas (s)",
        yaxis_title='Proporcja energii',
        yaxis=dict(range=[0, 1]),
        template='plotly_white',
        legend=dict(x=0.01, y=0.99),
        height=450
    )

    return fig



def plot_spectrum(frequencies, spectrum):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frequencies,
                            y=np.abs(spectrum), 
                            mode='lines', 
                            name='Widmo',
                            hovertemplate=
                                "<b>Częstotliwość:</b> %{x:.3f} s" + "<br>" +
                                "<b>Amplituda:</b> %{y:.3f}" + "<extra></extra>", 
                            ))
    fig.update_layout(
        xaxis_title='Częstotliwość (Hz)',
        yaxis_title='Amplituda',
        hovermode="closest"
    )



    return fig



def plot_cepstrum(cepstrum, sr):
    quefrency = np.arange(0, len(cepstrum)) / sr 

    min_pitch = 50
    max_pitch = 400
    min_quef = 1 / max_pitch
    max_quef = 1 / min_pitch

    valid_range = (quefrency > min_quef) & (quefrency < max_quef)
    pitch_quef = quefrency[valid_range]
    pitch_region = cepstrum[valid_range]

    peak_idx = np.argmax(pitch_region)
    peak_quef = pitch_quef[peak_idx]
    pitch_freq = 1 / peak_quef 

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=quefrency, 
                             y=cepstrum, 
                             mode='lines', 
                             name='Cepstrum',
                             hovertemplate=
                                "<b>Quefrency:</b> %{x:.3f} s" + "<br>" +
                                "<b>Amplituda:</b> %{y:.3f}" + "<extra></extra>"
                                ))
    fig.add_trace(go.Scatter(
        x=[peak_quef], y=[cepstrum[valid_range][peak_idx]],
        mode='markers+text',
        name=f'Pitch: {pitch_freq:.2f} Hz',
        text=[f'{pitch_freq:.1f} Hz'],
        textposition="top center",
        marker=dict(color='red', size=8),
        hovertemplate=
            "<b>Quefrency:</b> %{x:.3f} s" + "<br>" +
            "<b>Amplituda:</b> %{y:.3f}" + "<br>" +
            f"<b>Pitch:</b> {pitch_freq:.3f} Hz" + "<extra></extra>"
    ))

    fig.update_layout(
        xaxis=dict(range=[min_quef, max_quef]),
        yaxis=dict(range=[np.min(cepstrum[valid_range]) - 0.02, np.max(cepstrum[valid_range]) + 0.02]),
        xaxis_title="Quefrency (s)",
        yaxis_title="Amplituda",
        template="plotly_white"
    )

    return fig