import numpy as np
import pandas as pd
from scipy.signal import hilbert

import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks

# Feature extraction function: conserve les 24 premières features,
# puis ajoute des features SUPPLÉMENTAIRES classées par domaine.
def feature_extraction(data, fs=1.0):  # data (9999,1) ou (N,)
    def fft_fft(x):
        fft_trans = np.abs(np.fft.fft(x))
        freq_spectrum = fft_trans[1:int(np.floor(len(x) * 1.0 / 2)) + 1]
        _freq_sum_ = np.sum(freq_spectrum)
        return freq_spectrum, _freq_sum_

    data = np.asarray(data).ravel()
    freq_spectrum, _freq_sum_ = fft_fft(data)

    def signal_envelop(x):
        analytic_signal = hilbert(x)
        return np.abs(analytic_signal)

    # ---------- BASE (inchangé) ----------
    # Enveloppe
    signal_env = signal_envelop(data)

    # maximum / minimum
    dif_max = max(data)
    dif_min = min(data)
    # peak-to-peak
    dif_pk = int(dif_max) - int(dif_min)
    # moyenne
    dif_mean = data.mean()

    # médiane / skew / kurt
    dif_median = pd.Series(data).median()
    dif_skew = pd.Series(data).skew()
    dif_kurt = pd.Series(data).kurt()

    # skew/kurt enveloppe
    dif_skew_env = pd.Series(signal_env).skew()
    dif_kurt_env = pd.Series(signal_env).kurt()

    # spectre (stats simples)
    dif_meanFFT = freq_spectrum.mean()
    dif_maxFFT  = freq_spectrum.max()
    dif_varFFT  = freq_spectrum.var()
    dif_minFFT  = freq_spectrum.min()

    # variance / std
    dif_var = data.var()
    dif_std = data.std()

    # énergie (définition de ton code)
    dif_energy = np.sum(freq_spectrum ** 2) / len(freq_spectrum)

    # RMS (définition de ton code)
    dif_rms = np.sqrt(pow(dif_mean, 2) + pow(dif_std, 2))

    # ARV / facteurs waveform
    dif_arv = abs(data).mean()
    dif_boxing = dif_rms / (abs(data).mean() + 1e-12)
    dif_maichong = (max(data)) / (abs(data).mean() + 1e-12)
    dif_fengzhi = (max(data)) / (dif_rms + 1e-12)
    _sum = np.sum(np.sqrt(np.abs(data)))
    dif_yudu = max(data) / pow(_sum / (len(data) + 1e-12), 2)
    dif_qiaodu = (np.sum(data**4) / len(data)) / pow(dif_rms + 1e-12, 4)

    # entropie spectrale (définition de ton code)
    pr_freq = freq_spectrum * 1.0 / (_freq_sum_ + 1e-12)
    dif_entropy = -1 * np.sum([np.log2(p + 1e-5) * p for p in pr_freq])

    # ---------- LISTES PAR DOMAINE ----------
    # Temps (11) — inchangé
    time_features = [
        round(dif_max, 3), round(dif_min, 3), round(dif_pk, 3),
        round(dif_mean, 3), round(dif_median, 3), round(dif_var, 3),
        round(dif_std, 3), round(dif_skew, 3), round(dif_kurt, 3),
        round(dif_rms, 3), round(dif_energy, 3)
    ]

    # Waveform (8) — inchangé
    waveform_features = [
        round(dif_arv, 3), round(dif_boxing, 3), round(dif_maichong, 3),
        round(dif_fengzhi, 3), round(dif_yudu, 3), round(dif_qiaodu, 3),
        round(dif_skew_env, 3), round(dif_kurt_env, 3)
    ]

    # Spectral (5) — inchangé
    spectral_features = [
        round(dif_meanFFT, 3), round(dif_varFFT, 3),
        round(dif_maxFFT, 3),  round(dif_minFFT, 3),
        round(dif_entropy, 3)
    ]

    # ---------- AJOUTS DANS CHAQUE DOMAINE ----------
    # -- Temporel (ajouts) --
    duration = len(data) / fs if fs else float(len(data))
    zcr = np.mean(np.abs(np.diff(np.signbit(data))).astype(float))
    env_max_over_mean = np.max(signal_env) / (np.mean(signal_env) + 1e-12)
    env_max_over_median = np.max(signal_env) / (np.median(signal_env) + 1e-12)

    time_features_extra = [
        round(duration, 6),
        round(zcr, 6),
        round(env_max_over_mean, 6),
        round(env_max_over_median, 6),
    ]

    # -- Waveform (ajouts) --
    thr_env = 0.75 * np.max(signal_env) if signal_env.size else 0.0
    env_peaks, _ = find_peaks(signal_env, height=thr_env)
    n_env_peaks = len(env_peaks)
    mean_env_peaks = np.mean(signal_env[env_peaks]) if n_env_peaks > 0 else 0.0

    waveform_features_extra = [
        float(n_env_peaks),
        round(mean_env_peaks, 6),
    ]

    # -- Spectral (ajouts) --
    n = len(data)
    freqs = np.fft.fftfreq(n, d=1.0/fs)[1:int(np.floor(n/2)) + 1] if fs else np.arange(1, 1 + len(freq_spectrum))
    if len(freqs) == 0:
        freqs = np.arange(len(freq_spectrum), dtype=float)

    w = freq_spectrum + 1e-12
    centroid = np.sum(freqs * w) / np.sum(w)
    bw = np.sqrt(np.sum(((freqs - centroid) ** 2) * w) / np.sum(w))
    sigma = np.sqrt(np.sum(((freqs - centroid) ** 2) * w) / np.sum(w) + 1e-18)
    spec_skew = np.sum(((freqs - centroid) ** 3) * w) / (np.sum(w) * sigma**3 + 1e-18)
    spec_kurt = np.sum(((freqs - centroid) ** 4) * w) / (np.sum(w) * sigma**4 + 1e-18)
    p_safe = np.maximum(freq_spectrum, 1e-12)
    spec_flatness = np.exp(np.mean(np.log(p_safe))) / np.mean(p_safe)
    x = freqs - np.mean(freqs)
    y = freq_spectrum - np.mean(freq_spectrum)
    spec_slope = np.sum(x * y) / (np.sum(x**2) + 1e-12)
    cumsum = np.cumsum(freq_spectrum)
    roll85_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1])) if cumsum.size else 0
    roll95_idx = int(np.searchsorted(cumsum, 0.95 * cumsum[-1])) if cumsum.size else 0
    rolloff85 = freqs[roll85_idx] if roll85_idx < len(freqs) else (freqs[-1] if len(freqs) else 0.0)
    rolloff95 = freqs[roll95_idx] if roll95_idx < len(freqs) else (freqs[-1] if len(freqs) else 0.0)
    medianFFT = np.median(freq_spectrum) if freq_spectrum.size else 0.0
    peak_idx = int(np.argmax(freq_spectrum)) if freq_spectrum.size else 0
    peak_freq = freqs[peak_idx] if len(freqs) else float(peak_idx)
    thr_spec = 0.75 * np.max(freq_spectrum) if freq_spectrum.size else 0.0
    spec_peaks, _ = find_peaks(freq_spectrum, height=thr_spec)
    n_spec_peaks = len(spec_peaks)
    mean_spec_peaks = np.mean(freq_spectrum[spec_peaks]) if n_spec_peaks > 0 else 0.0

    spectral_features_extra = [
        round(medianFFT, 6),
        round(centroid, 6),
        round(bw, 6),
        round(spec_skew, 6),
        round(spec_kurt, 6),
        round(spec_flatness, 6),
        round(spec_slope, 6),
        round(rolloff85, 6),
        round(rolloff95, 6),
        round(peak_freq, 6),
        float(n_spec_peaks),
        round(mean_spec_peaks, 6),
    ]

    # ---------- Concaténation finale ----------
    # Ordre : Temps(11 + extra) + Waveform(8 + extra) + Spectral(5 + extra)
    time_all = time_features + time_features_extra
    waveform_all = waveform_features + waveform_features_extra
    spectral_all = spectral_features + spectral_features_extra

    feature_list = np.array(time_all + waveform_all + spectral_all, dtype=float)
    return feature_list


##### 24 Selected features using ANOVA

import numpy as np
import pandas as pd
from scipy.signal import hilbert, find_peaks

def feature_extraction_selected24(data, fs=1.0):
    """
    Calcule uniquement les 24 features sélectionnées
    (issues des domaines temporel, forme d’onde et spectral).
    Retourne:
        features_24 : np.ndarray de 24 valeurs
        names_24 : liste des noms correspondants
    """
    data = np.asarray(data).ravel()
    n = len(data)
    # --- FFT ---
    fft_mag = np.abs(np.fft.fft(data))
    freq_spectrum = fft_mag[1:int(np.floor(n / 2.0)) + 1]
    _freq_sum_ = np.sum(freq_spectrum) + 1e-12

    # --- Enveloppe ---
    analytic_signal = hilbert(data)
    signal_env = np.abs(analytic_signal)

    # --- Pré-calculs utiles ---
    dif_max = np.max(data)
    dif_min = np.min(data)
    dif_pk = int(dif_max) - int(dif_min)
    dif_var = np.var(data)
    dif_std = np.std(data)
    dif_mean = np.mean(data)
    dif_skew = pd.Series(data).skew()
    dif_kurt = pd.Series(data).kurt()
    dif_rms = np.sqrt(dif_mean**2 + dif_std**2)
    dif_arv = np.mean(np.abs(data))
    dif_boxing = dif_rms / (dif_arv + 1e-12)
    dif_maichong = dif_max / (dif_arv + 1e-12)
    dif_fengzhi = dif_max / (dif_rms + 1e-12)
    _sum_sqrt = np.sum(np.sqrt(np.abs(data)))
    dif_yudu = dif_max / ((_sum_sqrt / (n + 1e-12))**2)
    dif_qiaodu = (np.sum(data**4) / n) / (dif_rms + 1e-12)**4
    dif_skew_env = pd.Series(signal_env).skew()
    dif_kurt_env = pd.Series(signal_env).kurt()
    # --- Spectre ---
    dif_meanFFT = np.mean(freq_spectrum)
    dif_maxFFT = np.max(freq_spectrum)
    pr_freq = freq_spectrum / _freq_sum_
    dif_entropy = -np.sum(np.log2(pr_freq + 1e-5) * pr_freq)
    # --- Nouvelles features ajoutées ---
    zcr = np.mean(np.abs(np.diff(np.signbit(data))).astype(float))
    env_max_over_mean = np.max(signal_env) / (np.mean(signal_env) + 1e-12)
    # --- Spectral avancé ---
    freqs = np.fft.fftfreq(n, d=1.0/fs)[1:int(np.floor(n / 2.0)) + 1]
    w = freq_spectrum + 1e-12
    centroid = np.sum(freqs * w) / np.sum(w)
    bw = np.sqrt(np.sum(((freqs - centroid)**2) * w) / np.sum(w))
    sigma = np.sqrt(np.sum(((freqs - centroid)**2) * w) / np.sum(w) + 1e-18)
    spec_skew = np.sum(((freqs - centroid)**3) * w) / (np.sum(w) * sigma**3 + 1e-18)
    spec_kurt = np.sum(((freqs - centroid)**4) * w) / (np.sum(w) * sigma**4 + 1e-18)
    p_safe = np.maximum(freq_spectrum, 1e-12)
    spec_flatness = np.exp(np.mean(np.log(p_safe))) / np.mean(p_safe)
    x = freqs - np.mean(freqs)
    y = freq_spectrum - np.mean(freq_spectrum)
    spec_slope = np.sum(x * y) / (np.sum(x**2) + 1e-12)
    cumsum = np.cumsum(freq_spectrum)
    roll85_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    roll95_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
    rolloff85 = freqs[roll85_idx] if roll85_idx < len(freqs) else freqs[-1]
    rolloff95 = freqs[roll95_idx] if roll95_idx < len(freqs) else freqs[-1]

    # === 24 features sélectionnées ===
    features_24 = np.array([
        # Temporel
        round(dif_max, 3),          # 0 x_max
        round(dif_pk, 3),           # 2 peak_to_peak
        round(dif_std, 3),          # 6 std
        round(dif_skew, 3),         # 7 skewness
        round(dif_kurt, 3),         # 8 kurtosis
        round(zcr, 6),              # 13 zero_crossing_rate
        round(env_max_over_mean, 6),# 14 env_max_over_mean

        # Forme d’onde
        round(dif_boxing, 3),       # 17 waveform_factor
        round(dif_maichong, 3),     # 18 impulse_factor
        round(dif_fengzhi, 3),      # 19 crest_factor
        round(dif_qiaodu, 3),       # 21 kurtosis_factor
        round(dif_skew_env, 3),     # 22 envelope_skewness
        round(dif_kurt_env, 3),     # 23 envelope_kurtosis

        # Spectral
        round(dif_meanFFT, 3),      # 25 mean_spectrum
        round(dif_maxFFT, 3),       # 27 max_spectrum
        round(dif_entropy, 3),      # 29 spectral_entropy
        round(centroid, 6),         # 31 spectral_centroid
        round(bw, 6),               # 32 spectral_bandwidth
        round(spec_skew, 6),        # 33 spectral_skewness
        round(spec_kurt, 6),        # 34 spectral_kurtosis
        round(spec_flatness, 6),    # 35 spectral_flatness
        round(spec_slope, 6),       # 36 spectral_slope
        round(rolloff85, 6),        # 37 rolloff_85pct
        round(rolloff95, 6)         # 38 rolloff_95pct
    ], dtype=float)

    names_24 = [
        # Temporel
        "x_max","peak_to_peak","std","skewness","kurtosis",
        "zero_crossing_rate","env_max_over_mean",
        # Forme d’onde
        "waveform_factor","impulse_factor","crest_factor",
        "kurtosis_factor","envelope_skewness","envelope_kurtosis",
        # Spectral
        "mean_spectrum","max_spectrum","spectral_entropy",
        "spectral_centroid","spectral_bandwidth","spectral_skewness",
        "spectral_kurtosis","spectral_flatness","spectral_slope",
        "rolloff_85pct","rolloff_95pct"
    ]

    return features_24 #, names_24



# Feature extraction function, input data: difference (9999,1), original data (10000,1). Return (16,1).


# def feature_extraction(data):  # data (9999,1)
#     def fft_fft(data):
#         fft_trans = np.abs(np.fft.fft(data))
#         freq_spectrum = fft_trans[1:int(np.floor(len(data) * 1.0 / 2)) + 1]
#         _freq_sum_ = np.sum(freq_spectrum)
#         return freq_spectrum, _freq_sum_
#     freq_spectrum, _freq_sum_ = fft_fft(data)
    
#     def signal_envelop(data) :
#         analytic_signal = hilbert(data)
#         envelope = np.abs(analytic_signal)
        
#         return envelope
        
#     # maximum values
#     dif_max = max(data)
#     # minimum value
#     dif_min = min(data)
#     # peak-to-peak ratio
#     dif_pk = int(dif_max)-int(dif_min)
#     # average value
#     dif_mean = data.mean()
    
#     # median 
#     dif_median = pd.Series(data).median()
#     # Skewness
#     dif_skew = pd.Series(data).skew()
#     # skewness env
#     signal_env = signal_envelop(data)
#     dif_skew_env = pd.Series(signal_env).skew()
#     # kurtosis env 
#     dif_kurt_env = pd.Series(signal_env).kurt()
#     dif_meanFFT = freq_spectrum.mean()
#     dif_maxFFT = freq_spectrum.max()
#     dif_varFFT = freq_spectrum.var()
#     dif_minFFT = freq_spectrum.min()
    
    
    
#     # variance (statistics)
#     dif_var = data.var()
#     # (statistics) standard deviation
#     dif_std = data.std()
#     # energies
#     dif_energy = np.sum(freq_spectrum ** 2) / len(freq_spectrum)
#     # root mean square (math.)
#     dif_rms = np.sqrt(pow(dif_mean, 2) + pow(dif_std, 2))
#     # Rectification average
#     dif_arv = abs(data).mean()
#     # waveform factor
#     dif_boxing = dif_rms / (abs(data).mean())
#     # impulse factor  if
#     dif_maichong = (max(data)) / (abs(data).mean())
#     # crest factor   cf
#     dif_fengzhi = (max(data)) / dif_rms
#     # margin factor   cl
#     sum = 0
#     for i in range(len(data)):
#         sum += np.sqrt(abs(data[i]))
#     dif_yudu = max(data) / pow(sum / (len(data)), 2)
#     # kurtosis factor
#     dif_kurt = pd.Series(data).kurt()
#     # crag factor
#     dif_qiaodu = (np.sum([x ** 4 for x in data]) / len(data)) / pow(dif_rms, 4)
#     # information entropy
#     pr_freq = freq_spectrum * 1.0 / _freq_sum_
#     dif_entropy = -1 * np.sum([np.log2(p + 1e-5) * p for p in pr_freq])

#     # --- Time-domain (11) ---
#     # xmax, xmin, peak-to-peak, mean, median, variance, std, skewness, kurtosis, RMS, energy
#     time_features = [
#         round(dif_max, 3),       # x_max
#         round(dif_min, 3),       # x_min
#         round(dif_pk, 3),        # peak-to-peak
#         round(dif_mean, 3),      # mean
#         round(dif_median, 3),    # median
#         round(dif_var, 3),       # variance
#         round(dif_std, 3),       # std. deviation
#         round(dif_skew, 3),      # skewness
#         round(dif_kurt, 3),      # kurtosis
#         round(dif_rms, 3),       # RMS
#         round(dif_energy, 3),    # energy (time-domain)
#     ]
    
#     # --- Waveform-based (8) ---
#     # rectified mean, waveform factor, impulse factor, crest factor, margin factor,
#     # kurtosis factor, envelope skewness, envelope kurtosis
#     waveform_features = [
#         round(dif_arv, 3),         # rectified mean (ARV)
#         round(dif_boxing, 3),      # waveform factor
#         round(dif_maichong, 3),    # impulse factor
#         round(dif_fengzhi, 3),     # crest factor
#         round(dif_yudu, 3),        # margin factor
#         round(dif_qiaodu, 3),      # kurtosis factor
#         round(dif_skew_env, 3),    # envelope skewness
#         round(dif_kurt_env, 3),    # envelope kurtosis
#     ]
    
#     # --- Spectral-domain (5) ---
#     # mean spectrum, variance spectrum, max spectrum, min spectrum, spectral entropy
#     # (si tu as aussi l'énergie spectrale, insère-la avant 'entropy')
#     spectral_features = [
#         round(dif_meanFFT, 3),   # mean spectrum
#         round(dif_varFFT, 3),    # variance spectrum
#         round(dif_maxFFT, 3),    # max spectrum
#         round(dif_minFFT, 3),    # min spectrum
#         round(dif_entropy, 3),   # spectral entropy
#     ]
    
#     # --- Liste finale (24) ---
#     # feature_list = time_features + waveform_features + spectral_features

#     feature_list = np.array([round(dif_max, 3),       # x_max
#     round(dif_min, 3),       # x_min
#     round(dif_pk, 3),        # peak-to-peak
#     round(dif_mean, 3),      # mean
#     round(dif_median, 3),    # median
#     round(dif_var, 3),       # variance
#     round(dif_std, 3),       # std. deviation
#     round(dif_skew, 3),      # skewness
#     round(dif_kurt, 3),      # kurtosis
#     round(dif_rms, 3),       # RMS
#     round(dif_energy, 3),    # energy (time-domain))
#     round(dif_arv, 3),         # rectified mean (ARV)
#     round(dif_boxing, 3),      # waveform factor
#     round(dif_maichong, 3),    # impulse factor
#     round(dif_fengzhi, 3),     # crest factor
#     round(dif_yudu, 3),        # margin factor
#     round(dif_qiaodu, 3),      # kurtosis factor
#     round(dif_skew_env, 3),    # envelope skewness
#     round(dif_kurt_env, 3),    # envelope kurtosis
#     round(dif_meanFFT, 3),   # mean spectrum
#     round(dif_varFFT, 3),    # variance spectrum
#     round(dif_maxFFT, 3),    # max spectrum
#     round(dif_minFFT, 3),    # min spectrum
#     round(dif_entropy, 3)])   # spectral entropy
    
    
#     return feature_list

# import scipy.io as scio
# rootpath = 'C:/Users/michel/Downloads/Das_data'
# train_rootpath = r"C:\Users\michel\Downloads\Das_data\train\01_background\220112_cxm_background_01_single_data_1.mat" #rootpath+'/train"
# path = train_rootpath #train_rootpath + name_list[i].split(' ')[0]
# data = scio.loadmat(path)['data'][:,1]

# def signal_envelop(data) :
#     analytic_signal = hilbert(data)
#     envelope = np.abs(analytic_signal)
#     return envelope
# # median 

# dif_median = pd.Series(data).median()
# # Skewness
# dif_skew = pd.Series(data).skew()
# # skewness env
# signal_env = signal_envelop(data)
# dif_skew_env = pd.Series(signal_env).skew()
# # kurtosis env 
# dif_kurt_env = pd.Series(signal_env).kurt()
