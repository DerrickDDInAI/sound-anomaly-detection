"""
Functions related to sounds:
- get the sound filepaths and metadata
- extract the sound features
"""
# =====================================================================
# Import
# =====================================================================

# Import internal modules
import os.path
from typing import List, Set, Dict, TypedDict, Tuple, Optional

# Import 3rd party modules
import pandas as pd
import numpy as np
import librosa
import librosa.display

# ============================================================
# Functions
# ============================================================


def get_sound_files(folder: str) -> pd.DataFrame:
    """
    Function to get all sound files within a folder.
    Param: `folder` name includes background noise level and machine type (e.g. -6_db_fan)    
    Return: a DataFrame  
    """
    # Create empty lists
    sound_list: List[str] = []  # will contain all sound filenames
    sound_path_list: List[str] = []  # will contain the sound's filepaths
    # will contain the background noise level in db of each sound
    noise_db_list: List[int] = []
    # will contain the machine type of each sound
    machine_type_list: List[str] = []
    # will contain the machine product id of each sound
    model_id_list: List[str] = []
    # will contain the target value (normal=0 or anormal=1) for each sound
    target_list: List[str] = []

    # Get the relative path of the directory that contains all the sound files
    folder_path = os.path.join("assets", "sounds", folder)

    # Get all the filenames within the directory
    for path, dirs, files in os.walk(folder_path):
        for filename in files:
            # Search only filenames with the extension ".wav"
            if filename.lower().endswith(".wav"):

                # Get the filename
                sound_list.append(filename)

                # Get the filepath
                sound_path = os.path.join(path, filename)
                sound_path_list.append(sound_path)

                # Split filepath to retrieve the information
                path_splitted = sound_path.split("/")

                # Get the background noise in db
                noise_db = int(path_splitted[2].split("_")[0])
                noise_db_list.append(noise_db)

                # Get the machine type
                machine_type = path_splitted[2].split("_")[2]
                machine_type_list.append(machine_type)

                # Get the model id
                model_id = path_splitted[3].split("_")[1]
                model_id_list.append(model_id)

                # Get target variable (normal or anormal)
                target = path_splitted[4]
                target_list.append(target)

    # Create list with the data
    data: List[float, int, str] = list(zip(
        noise_db_list, machine_type_list, model_id_list, sound_list, sound_path_list, target_list))

    # Create list with column names
    cols: List[str] = ["noise_db", "machine_type",
                       "model_id", "sound", "sound_path", "target"]

    # Return a DataFrame from the lists
    return pd.DataFrame(data=data, columns=cols)


def get_all_sounds(db_list: List[int], machine_type_list: List[str]) -> pd.DataFrame:
    """
    Function to get all sound files for specified lists of background noise and machine type.
    Param: * `db_list` is a list of background noise level (i.e. -6, 0, 6)
           * `machine_type_list` is a list of machine type (i.e. fan, pump, valve, slider) 
    Return: a DataFrame  
    """
    df_list = []
    for db in db_list:
        for machine_type in machine_type_list:
            df = get_sound_files(f"{db}_dB_{machine_type}")
            df_list.append(df)

    return pd.concat(df_list, axis=0)


def get_audio_features(sound_path: str) -> List[float]:
    """
    Function to extract audio features from the sound file
    Param: `sound_path` is the filepath of the sound
    Return: a list of audio features aggregated by their average values
    """
    # Read audio file
    y, sr = librosa.load(sound_path)

    # short-time Fourier Transform
    stft = librosa.stft(y)

    # Get spectogram
    spect: np.ndarray = np.abs(stft)
    spect_mean: np.float = np.mean(spect)
    spect_min: np.float = np.min(spect)
    spect_max: np.float = np.max(spect)
    spect_std: np.float = np.std(spect)

    # Get mel spectogram
    mel_spect: np.ndarray = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048)
    mel_spect_mean: np.float = np.mean(mel_spect)
    mel_spect_min: np.float = np.min(mel_spect)
    mel_spect_max: np.float = np.max(mel_spect)
    mel_spect_std: np.float = np.std(mel_spect)

    # Get chromagram
    chroma: np.ndarray = librosa.feature.chroma_stft(S=spect, sr=sr)
    chroma_mean: np.float = np.mean(chroma)
    chroma_min: np.float = np.min(chroma)
    chroma_max: np.float = np.max(chroma)
    chroma_std: np.float = np.std(chroma)

    # Get constant-Q chromagram
    chroma_cq: np.ndarray = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cq_mean: np.float = np.mean(chroma_cq)
    chroma_cq_min: np.float = np.min(chroma_cq)
    chroma_cq_max: np.float = np.max(chroma_cq)
    chroma_cq_std: np.float = np.std(chroma_cq)

    # Get chromagram cens
    chroma_cens: np.ndarray = librosa.feature.chroma_cens(y=y, sr=sr)
    chroma_cens_mean: np.float = np.mean(chroma_cens)
    chroma_cens_min: np.float = np.min(chroma_cens)
    chroma_cens_max: np.float = np.max(chroma_cens)
    chroma_cens_std: np.float = np.std(chroma_cens)

    # Get mfcc
    mfcc: np.ndarray = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean: np.float = np.mean(mfcc)
    mfcc_min: np.float = np.min(mfcc)
    mfcc_max: np.float = np.max(mfcc)
    mfcc_std: np.float = np.std(mfcc)

    # Get rms
    S: np.ndarray
    phase: np.ndarray
    S, phase = librosa.magphase(stft)
    rms: np.ndarray = librosa.feature.rms(S=S)
    rms_mean: np.float = np.mean(rms)
    rms_min: np.float = np.min(rms)
    rms_max: np.float = np.max(rms)
    rms_std: np.float = np.std(rms)

    # Get spectral centroid
    cent: np.ndarray = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean: np.float = np.mean(cent)
    cent_min: np.float = np.min(cent)
    cent_max: np.float = np.max(cent)
    cent_std: np.float = np.std(cent)

    # Get spectral bandwidth
    spec_bw: np.ndarray = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bw_mean: np.float = np.mean(spec_bw)
    spec_bw_min: np.float = np.min(spec_bw)
    spec_bw_max: np.float = np.max(spec_bw)
    spec_bw_std: np.float = np.std(spec_bw)

    # Get spectral contrast
    contrast: np.ndarray = librosa.feature.spectral_contrast(S=S, sr=sr)
    contrast_mean: np.float = np.mean(contrast)
    contrast_min: np.float = np.min(contrast)
    contrast_max: np.float = np.max(contrast)
    contrast_std: np.float = np.std(contrast)

    # Get spectral flatness
    flatness: np.ndarray = librosa.feature.spectral_flatness(y=y)
    flatness_mean: np.float = np.mean(flatness)
    flatness_min: np.float = np.min(flatness)
    flatness_max: np.float = np.max(flatness)
    flatness_std: np.float = np.std(flatness)

    # Get roll-off frequency
    roll_off: np.ndarray = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.95)
    roll_off_mean: np.float = np.mean(roll_off)
    roll_off_min: np.float = np.min(roll_off)
    roll_off_max: np.float = np.max(roll_off)
    roll_off_std: np.float = np.std(roll_off)

    # tonal centroid features (tonnetz)
    tonnetz: np.ndarray = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean: np.float = np.mean(tonnetz)
    tonnetz_min: np.float = np.min(tonnetz)
    tonnetz_max: np.float = np.max(tonnetz)
    tonnetz_std: np.float = np.std(tonnetz)

    # zero-crossing rate
    zero_crossing_rate: np.ndarray = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean: np.float = np.mean(zero_crossing_rate)
    zero_crossing_rate_min: np.float = np.min(zero_crossing_rate)
    zero_crossing_rate_max: np.float = np.max(zero_crossing_rate)
    zero_crossing_rate_std: np.float = np.std(zero_crossing_rate)

    # d_harmonic, d_percussive
    d_harmonic, d_percussive = librosa.decompose.hpss(stft)
    d_harmonic_abs = np.abs(d_harmonic)
    d_percussive_abs = np.abs(d_percussive)

    d_harmonic_mean: np.float = np.mean(d_harmonic_abs)
    d_harmonic_min: np.float = np.min(d_harmonic_abs)
    d_harmonic_max: np.float = np.max(d_harmonic_abs)
    d_harmonic_std: np.float = np.std(d_harmonic_abs)

    d_percussive_mean: np.float = np.mean(d_percussive_abs)
    d_percussive_min: np.float = np.min(d_percussive_abs)
    d_percussive_max: np.float = np.max(d_percussive_abs)
    d_percussive_std: np.float = np.std(d_percussive_abs)

    # Return a list of audio features aggregated by their average values
    return [spect_mean, spect_min, spect_max, spect_std,
            mel_spect_mean, mel_spect_min, mel_spect_max, mel_spect_std,
            chroma_mean, chroma_min, chroma_max, chroma_std,
            chroma_cq_mean, chroma_cq_min, chroma_cq_max, chroma_cq_std,
            chroma_cens_mean, chroma_cens_min, chroma_cens_max, chroma_cens_std,
            mfcc_mean, mfcc_min, mfcc_max, mfcc_std,
            rms_mean, rms_min, rms_max, rms_std,
            cent_mean, cent_min, cent_max, cent_std,
            spec_bw_mean, spec_bw_min, spec_bw_max, spec_bw_std,
            contrast_mean, contrast_min, contrast_max, contrast_std,
            flatness_mean, flatness_min, flatness_max, flatness_std,
            roll_off_mean, roll_off_min, roll_off_max, roll_off_std,
            tonnetz_mean, tonnetz_min, tonnetz_max, tonnetz_std,
            zero_crossing_rate_mean, zero_crossing_rate_min, zero_crossing_rate_max, zero_crossing_rate_std,
            d_harmonic_mean, d_harmonic_min, d_harmonic_max, d_harmonic_std,
            d_percussive_mean, d_percussive_min, d_percussive_max, d_percussive_std
            ]


# List of audio feature names to use as column
audio_features_cols: List[str] = ["spect_mean", "spect_min", "spect_max", "spect_std",
                                  "mel_spect_mean", "mel_spect_min", "mel_spect_max", "mel_spect_std",
                                  "chroma_mean", "chroma_min", "chroma_max", "chroma_std",
                                  "chroma_cq_mean", "chroma_cq_min", "chroma_cq_max", "chroma_cq_std",
                                  "chroma_cens_mean", "chroma_cens_min", "chroma_cens_max", "chroma_cens_std",
                                  "mfcc_mean", "mfcc_min", "mfcc_max", "mfcc_std",
                                  "rms_mean", "rms_min", "rms_max", "rms_std",
                                  "cent_mean", "cent_min", "cent_max", "cent_std",
                                  "spec_bw_mean", "spec_bw_min", "spec_bw_max", "spec_bw_std",
                                  "contrast_mean", "contrast_min", "contrast_max", "contrast_std",
                                  "flatness_mean", "flatness_min", "flatness_max", "flatness_std",
                                  "roll_off_mean", "roll_off_min", "roll_off_max", "roll_off_std",
                                  "tonnetz_mean", "tonnetz_min", "tonnetz_max", "tonnetz_std",
                                  "zero_crossing_rate_mean", "zero_crossing_rate_min", "zero_crossing_rate_max", "zero_crossing_rate_std",
                                  "d_harmonic_mean", "d_harmonic_min", "d_harmonic_max", "d_harmonic_std",
                                  "d_percussive_mean", "d_percussive_min", "d_percussive_max", "d_percussive_std"
                                  ]
