import time
import streamlit as st
import librosa
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from audio_analysis import *

st.markdown(
    """
    <style>
        .main {
            max-width: 90% !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-right: 10;
            padding-left: 10;
            max-width: 90% !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Analiza sygnału audio")

uploaded_file = st.file_uploader("Wybierz plik WAV", type=["wav"])

clip_duration = None
frame_size = None
hop_length = None  

if uploaded_file is not None:
    st.audio(uploaded_file)
    y, sr = librosa.load(uploaded_file, sr=None)

    col1, col2 = st.columns([1, 3]) 


    with col1:
        st.write("### Opcje analizy:")
        analysis_type = st.radio("Wybierz typ analizy:", ('Cały sygnał', 'Konkretna ramka'), key="analysis_type_key")

        frame_duration = st.selectbox("Wybierz długość ramki:", ["10ms", "20ms", "30ms", "40ms"], key="frame_duration_2")
        frame_duration = int(frame_duration[:-2])  
        frame_size = int(frame_duration * sr / 1000)

        if analysis_type == "Konkretna ramka":
            start_time_ms = st.text_input("Czas startu ramki (ms):", "0")
            start_time_ms = int(start_time_ms)
            end_time_ms = st.text_input("Czas końca ramki (ms):", "0")
            end_time_ms = int(end_time_ms)

            if end_time_ms > len(y):
                st.warning("Ramka wychodzi poza długość sygnału!")
            elif end_time_ms < start_time_ms:
                st.warning("Czas końca ramki musi być większy od czasu startu!")
            elif start_time_ms < 0:
                st.warning("Czas startu ramki musi być większy od 0!")
            else:
                y = y[int(start_time_ms * sr / 1000): int(end_time_ms * sr / 1000)]
        # else:
        #     y, sr = librosa.load(uploaded_file, sr=None)


    with col2:
        st.write("### Wykres przebiegu czasowego audio:")
        if analysis_type == "Cały sygnał":
            time_axis = np.linspace(0, len(y) / sr, len(y))

        if analysis_type == "Konkretna ramka":
            time_axis = np.linspace(start_time_ms / 1000, end_time_ms / 1000, len(y))

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=time_axis,
                                y=y,
                                mode='lines', 
                                name="Przebieg czasowy", 
                                hovertemplate="<b>Czas:</b> %{x:.3f} s" + "<br>" +
                                            "<b>Amplituda:</b> %{y:.3f}" + "<extra></extra>", 
        ))
        fig.update_layout(
            title="Przebieg czasowy",
            title_x=0.5,
            xaxis_title="Czas (s)",
            yaxis_title="Amplituda",
            hovermode="closest"  
        )

        st.plotly_chart(fig)
        
    
    st.write("### Analiza w dziedzinie częstotliwości:")
    hop_length = frame_size // 2
    frame_features = extract_frame_features(y, sr, frame_size, hop_length)  

    features = [
        ('volume', frame_features['volume']),
        ('centroid', frame_features['centroid']),
        ('bandwidth', frame_features['bandwidth']),
        ('energy_ratios', frame_features['energy_ratios']),
        ('flatness', frame_features['flatness']),
        ('crest', frame_features['crest'])    
    ]

    num_features = len(features)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_feature(time_axis, features[0][1], features[0][0], "ramki"), use_container_width=True)
    with col2:
        st.plotly_chart(plot_feature(time_axis, features[1][1], features[1][0], "ramki"), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_feature(time_axis, features[2][1], features[2][0], "ramki"), use_container_width=True)
    with col4:
        st.plotly_chart(plot_feature(time_axis, features[5][1], features[5][0], "ramki"), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:  
        st.plotly_chart(plot_feature(time_axis, features[4][1], features[4][0], "ramki"), use_container_width=True)
    with col6:  
        st.plotly_chart(plot_ersb(time_axis, features[3][1], "ramki"), use_container_width=True)



    # if analysis_type == 'Klip':
    #     df = pd.DataFrame(frame_features)
    # else: 
    #     df = pd.DataFrame(frame_features)

    # csv_buffer = io.StringIO()
    # df.to_csv(csv_buffer, index=False)
    # csv_data = csv_buffer.getvalue()

    # st.write("### Podgląd danych CSV:")
    # st.dataframe(df.head(5))

#     st.download_button(
#         label="📥 Pobierz CSV",
#         data=csv_data,
#         file_name="audio_analysis.csv",
#         mime="text/csv"
# )
