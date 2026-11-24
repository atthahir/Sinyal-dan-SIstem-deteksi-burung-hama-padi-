#!/usr/bin/env python
"""
Topik 1 - Windowing & STFT pada suara burung hama padi
Analisis Pengaruh Ukuran dan Jenis Window STFT terhadap Representasi Spektral
dan Klasifikasi Suara Burung Hama Padi.

Jalankan dari folder bird_sounds/ (yang berisi training/ dan testing/):

    python topik1_stft.py                       # uji cepat (300 klip/kelas training, 100 testing)
    python topik1_stft.py --full                # data penuh (untuk hasil laporan)
    python topik1_stft.py --merge6              # gabungkan Bulbul & Chubbs_cis -> Other_Birds (6 kelas)

Semua tabel/grafik/konfigurasi disimpan di bird_sounds/hasil/.
Bagian kode diberi label A-F sesuai checklist kelompok.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # hemat memori: tiap proses paralel hanya 1 thread BLAS
import argparse, json, platform, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # simpan gambar ke file, tidak membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import librosa, librosa.display
import soundfile as sf
import scipy, sklearn
import scikit_posthocs as sp
from joblib import Parallel, delayed
from scipy import signal
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

WIN_SIZES = [256, 512, 1024, 2048]
WIN_TYPES = ["boxcar", "hann", "hamming", "blackman"]      # boxcar = rectangular
NICE = {"boxcar": "Rectangular", "hann": "Hann", "hamming": "Hamming", "blackman": "Blackman"}


# =============================================================================
# KONFIGURASI
# =============================================================================
def get_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(Path(__file__).resolve().parent), help="folder berisi training/ dan testing/")
    ap.add_argument("--sub-train", type=int, default=300, help="klip/kelas untuk CV (default 300)")
    ap.add_argument("--sub-test", type=int, default=100, help="klip/kelas untuk test akhir (default 100)")
    ap.add_argument("--full", action="store_true", help="pakai semua klip (abaikan sub-train/sub-test)")
    ap.add_argument("--merge6", action="store_true", help="Bulbul & Chubbs_cis digabung ke Other_Birds (6 kelas)")
    ap.add_argument("--sr", type=int, default=22050, help="sample rate target")
    ap.add_argument("--hop-ratio", type=float, default=0.5, help="hop = N * ratio (0.5 = overlap 50%%)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fir-taps", type=int, default=129)
    ap.add_argument("--jobs", type=int, default=1, help="jumlah proses paralel (default 1 = aman memori; naikkan ke 2-4 hanya jika RAM/virtual memory besar)")
    ap.add_argument("--max-dur", type=float, default=3.0, help="durasi maks per klip (detik); klip lebih panjang dipotong")
    ap.add_argument("--clear", action="store_true", help="bersihkan layar terminal saat mulai (menghilangkan echo perintah Run)")
    ap.add_argument("--seed", type=int, default=42, help="seed global")
    a = ap.parse_args()
    if a.clear:
        print("\033[2J\033[3J\033[H", end="")
    if a.full:
        a.sub_train = a.sub_test = None
    a.preemph, a.bp_low, a.bp_high = 0.97, 1000, 10000
    a.data_root = Path(a.data_root)
    a.out = a.data_root / "hasil"
    a.out.mkdir(exist_ok=True)
    return a


def parallel(cfg, tasks, batch_size="auto"):
    """Jalankan tugas joblib; jika proses paralel gagal (mis. memori Windows habis) ulangi dengan 1 proses."""
    if cfg.jobs == 1:
        return Parallel(n_jobs=1)(tasks)         # generator diproses satu per satu (hemat memori)
    tasks = list(tasks)
    try:
        return Parallel(n_jobs=cfg.jobs, batch_size=batch_size)(tasks)
    except Exception as e:
        print(f"  PERINGATAN: paralel gagal ({type(e).__name__}); mengulang dengan 1 proses (lebih lambat tapi aman).")
        return Parallel(n_jobs=1)(tasks)


def savefig(cfg, name):
    plt.tight_layout()
    plt.savefig(cfg.out / name, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [gambar] {name}")


def save_table(cfg, df, name, index=True):
    df.to_csv(cfg.out / name, index=index)
    print(f"  [tabel]  {name}")


def title(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# =============================================================================
# A. KESIAPAN DATA
# =============================================================================
def stratified_subsample(d, n, seed):
    """Ambil maksimal n klip per kelas (dipakai hanya untuk uji cepat; --full mematikan ini)."""
    if n is None:
        return d
    return pd.concat([g.sample(min(len(g), n), random_state=seed) for _, g in d.groupby("label")])


def label_from_filename(path):
    """
    CARA KODE MENENTUKAN KELAS.
    Kelas diambil dari NAMA FILE: teks sebelum titik pertama.
        'Noise.0001.wav'          -> 'Noise'
        'VillageWeaver.xyz.wav'   -> 'VillageWeaver'
    Folder training/ dan testing/ hanya menentukan SPLIT (latih atau uji), bukan kelasnya.
    Daftar kelas final = semua label unik yang ditemukan, diurutkan abjad (lihat A_data).
    """
    return Path(path).name.split(".")[0]


def scan_wavs(cfg):
    """Langkah 1-2: pindai semua .wav, baca header (fs, durasi, kanal) TANPA memuat sinyalnya."""
    rows, skipped = [], []
    for split in ("training", "testing"):                    # hanya dua folder ini (abaikan .venv, hasil, dll.)
        folder = cfg.data_root / split
        if not folder.is_dir():
            print(f"PERINGATAN: folder {folder} tidak ditemukan")
            continue
        for p in sorted(folder.rglob("*.wav")):
            try:
                info = sf.info(str(p))                       # membaca header saja -> cepat dan hemat memori
            except Exception as e:                           # file rusak: catat, lewati, jangan hentikan program
                skipped.append(dict(file=str(p), error=type(e).__name__)); continue
            rows.append(dict(path=str(p), split=split, label=label_from_filename(p),
                             sr=info.samplerate, dur=info.duration, ch=info.channels))
    return rows, skipped


def check_labels(df, cfg):
    """Langkah 3: validasi label supaya salah baca nama file ketahuan sejak awal."""
    lab_tr, lab_te = set(df[df.split == "training"].label), set(df[df.split == "testing"].label)
    if lab_tr != lab_te:
        print(f"PERINGATAN: kelas training dan testing berbeda! "
              f"hanya di training: {sorted(lab_tr - lab_te)}, hanya di testing: {sorted(lab_te - lab_tr)}")
    n_cls = df.label.nunique()
    if n_cls > 12:                                           # tanda aturan nama file tidak cocok (label = nama file utuh)
        raise SystemExit(f"Terdeteksi {n_cls} kelas, terlalu banyak. Kemungkinan nama file tidak berformat "
                         f"'Kelas.xxx.wav'. Contoh label terbaca: {sorted(df.label.unique())[:5]}")
    min_tr = df[df.split == "training"].label.value_counts().min()
    if min_tr < cfg.folds:                                   # StratifiedKFold butuh minimal 'folds' klip per kelas
        raise SystemExit(f"Kelas terkecil di training hanya {min_tr} klip, kurang dari {cfg.folds} fold.")


def make_folds(cfg, ctx):
    """
    CARA 5 FOLD DISUSUN (Stratified K-Fold, diulang untuk tiap seed).
    - Hanya klip TRAINING yang dibagi. Klip testing tidak disentuh sampai evaluasi akhir.
    - Stratified: proporsi tiap kelas dijaga sama di setiap fold.
    - Tiap seed mengacak ulang pembagiannya (seed 0, 1, 2 -> 3 pembagian berbeda).
    - Hasil disimpan ke A4 (fold tiap klip) dan A5 (jumlah klip per kelas per fold) sebagai bukti.
    """
    idx_tr = np.where(ctx["is_tr"])[0]
    ytr = ctx["y"][idx_tr]
    df, classes = ctx["df"], ctx["classes"]
    folds = {s: list(StratifiedKFold(cfg.folds, shuffle=True, random_state=s).split(idx_tr, ytr)) for s in cfg.seeds}

    assign = pd.DataFrame({"file": df.path.iloc[idx_tr].map(lambda p: Path(p).name).values,
                           "label": df.label.iloc[idx_tr].values})
    per_class = []
    for s, fl in folds.items():
        fold_of = np.full(len(idx_tr), -1)
        for f, (tr, va) in enumerate(fl):
            fold_of[va] = f
        # verifikasi: setiap klip training tepat satu kali menjadi data validasi
        assert (fold_of >= 0).all() and sum(len(va) for _, va in fl) == len(idx_tr), "fold tidak lengkap/tumpang tindih"
        assign[f"fold_seed{s}"] = fold_of
        ct = pd.crosstab(assign.label, fold_of); ct.columns = [f"fold{c}" for c in ct.columns]
        ct.insert(0, "seed", s); per_class.append(ct)
    pc = pd.concat(per_class)
    print(f"\nA3. {cfg.folds}-fold berstrata x {len(cfg.seeds)} seed disusun "
          f"({len(idx_tr)} klip training; {len(idx_tr)//cfg.folds} klip validasi per fold)")
    print(f"    Contoh seed {cfg.seeds[0]}: jumlah klip validasi per kelas di tiap fold\n" +
          pc[pc.seed == cfg.seeds[0]].drop(columns="seed").to_string())
    save_table(cfg, assign, "A4_fold_per_klip.csv", index=False)
    save_table(cfg, pc, "A5_fold_per_kelas.csv")
    ctx.update(folds=folds, idx_tr=idx_tr, ytr=ytr)


def A_data(cfg):
    title("A. KESIAPAN DATA")
    # 1-2) Baca semua file dan catat fs, durasi, kanal, label
    rows, skipped = scan_wavs(cfg)
    if skipped:
        print(f"PERINGATAN: {len(skipped)} file .wav tidak terbaca & dilewati, contoh: {[s['file'] for s in skipped[:3]]}")
        save_table(cfg, pd.DataFrame(skipped), "A0_file_rusak.csv", index=False)
    else:
        print("Pengecekan file: semua .wav terbaca (tidak ada file rusak).")
    if not rows:
        raise SystemExit(f"Tidak ada .wav di {cfg.data_root}. Pastikan training/ dan testing/ sudah diekstrak "
                         "(training.rar harus di-extract dulu).")
    df = pd.DataFrame(rows)
    if cfg.merge6:                                           # opsional: gabungkan label Bulbul & Chubbs_cis ke Other_Birds
        df["label"] = df["label"].replace({"Bulbul": "Other_Birds", "Chubbs_cis": "Other_Birds"})

    # 3) Validasi label hasil pembacaan nama file
    check_labels(df, cfg)

    # 4) Jumlah klip per kelas dan per split, serta fs per kelas
    print(f"Total klip: {len(df)}")
    ct = pd.crosstab(df["label"], df["split"], margins=True)
    print(ct.to_string())
    cnt = df.label.value_counts()
    print(f"\nKetimpangan kelas (terbanyak/tersedikit): {cnt.max()}/{cnt.min()} = {cnt.max()/cnt.min():.2f}x "
          f"-> SVM memakai class_weight='balanced' untuk mengimbangi.")
    print("\nSample rate (Hz):", df.sr.value_counts().to_dict())
    print("Channel         :", df.ch.value_counts().to_dict())
    print("Durasi (s)      :", {k: round(v, 3) for k, v in df.dur.describe().to_dict().items()})
    fs_ct = pd.crosstab(df["label"], df["sr"], margins=True)
    print("\nSample rate per kelas (jumlah klip)\n" + fs_ct.to_string())
    if df.sr.nunique() > 1 or (df.dur.max() - df.dur.min()) > 0.5:
        print(f"-> sample rate/durasi tidak seragam: semua klip di-resample ke {cfg.sr} Hz "
              f"(batas Nyquist baru = {cfg.sr/2:g} Hz).")
        print(f"   {int((df.dur > cfg.max_dur + 0.05).sum())} klip lebih panjang dari {cfg.max_dur:g} s -> dipotong ke {cfg.max_dur:g} s pertama; "
              f"{int((df.dur < 0.37).sum())} klip < 0,37 s -> diulang (tile) sampai 0,37 s.")
    save_table(cfg, ct, "A1_jumlah_klip.csv")
    save_table(cfg, df.drop(columns="path"), "A2_inventaris.csv", index=False)
    save_table(cfg, fs_ct, "A2b_fs_per_kelas.csv")

    # Daftar SEMUA file (jumlah baris = Total klip), dikelompokkan per kelas lalu per folder
    daftar = df.assign(file=df.path.map(lambda p: Path(p).name))[["label", "split", "file", "sr", "dur"]]
    daftar = daftar.sort_values(["label", "split", "file"]).reset_index(drop=True)
    daftar.insert(0, "no", np.arange(1, len(daftar) + 1))                # nomor urut 1..N
    daftar.insert(2, "kelas_id", daftar.label.map({c: i for i, c in enumerate(sorted(df.label.unique()))}))
    assert len(daftar) == len(df) == ct.loc["All", "All"], "jumlah baris daftar tidak sama dengan total klip"
    daftar.to_csv(cfg.out / "A1b_daftar_file.csv", index=False, encoding="utf-8-sig")   # utf-8-sig: aman dibuka di Excel
    print(f"  [tabel]  A1b_daftar_file.csv  ({len(daftar)} baris = total klip, urut per kelas)")
    kolom = {f"{lb}_{sp_} ({len(g)})": pd.Series(g.file.values)           # format lebar: 1 kolom = 1 kelas di 1 folder
             for (lb, sp_), g in daftar.groupby(["label", "split"])}
    pd.DataFrame(kolom).to_csv(cfg.out / "A1c_daftar_file_lebar.csv", index=False, encoding="utf-8-sig")
    print("  [tabel]  A1c_daftar_file_lebar.csv")

    # 5) Pilih klip yang dipakai (--full = semua) dan tetapkan daftar kelas
    tr = stratified_subsample(df[df.split == "training"], cfg.sub_train, cfg.seed).reset_index(drop=True)
    te = stratified_subsample(df[df.split == "testing"], cfg.sub_test, cfg.seed).reset_index(drop=True)
    if te.empty or tr.empty:
        raise SystemExit("Folder training/ atau testing/ kosong.")
    d = pd.concat([tr, te], ignore_index=True)
    classes = sorted(d.label.unique())                       # urutan abjad -> nomor kelas 0..5 tetap sama tiap run
    ctx = dict(df=d, classes=classes, bird=[c for c in classes if c != "Noise"],
               y=d.label.map({c: i for i, c in enumerate(classes)}).values,
               is_tr=(d.split == "training").values)
    print(f"\nDipakai: {len(tr)} klip training | {len(te)} klip testing | {len(classes)} kelas: {classes}")
    print("Pemetaan kelas -> nomor:", {c: i for i, c in enumerate(classes)})

    # 6) Susun 5 fold (berstrata, tiap seed)
    make_folds(cfg, ctx)
    return ctx


# =============================================================================
# B. PRAPROSES DAN KODE
# =============================================================================
def make_filters(cfg):
    sos = signal.butter(4, [cfg.bp_low, cfg.bp_high], btype="bandpass", fs=cfg.sr, output="sos")
    fir = signal.firwin(cfg.fir_taps, [cfg.bp_low, cfg.bp_high], pass_zero=False, fs=cfg.sr)
    return sos, fir


MIN_SAMPLES = 8192          # klip lebih pendek (~0,37 s) diulang/di-tile agar STFT N=2048 punya cukup frame


def load_preemph(path, sr, a, max_dur):
    y, _ = librosa.load(path, sr=sr, mono=True, duration=max_dur)   # langkah 1: resample (+ potong durasi)
    if len(y) < MIN_SAMPLES:
        y = np.tile(y, int(np.ceil(MIN_SAMPLES / max(len(y), 1))))[:MIN_SAMPLES]
    return np.append(y[0], y[1:] - a * y[:-1]).astype(np.float32)    # langkah 2: pre-emphasis


def apply_filter(y, kind, sos, fir):                                 # langkah 3: bandpass (zero-phase)
    if kind == "none":
        return y
    if kind == "iir":
        return signal.sosfiltfilt(sos, y).astype(np.float32)
    if kind == "fir":
        return signal.filtfilt(fir, [1.0], y).astype(np.float32)
    raise ValueError(kind)


def snr_db(y, N=1024, hop=512):
    """Perkiraan SNR tanpa referensi bersih: energi frame 20% teratas vs 20% terbawah (frame hening diabaikan)."""
    e = (np.abs(librosa.stft(y, n_fft=N, hop_length=hop))**2).sum(0)
    e = e[e > 1e-6 * e.max()]
    return 10*np.log10(e[e >= np.percentile(e, 80)].mean() / e[e <= np.percentile(e, 20)].mean())


def clip_features(y, sr, N, win, hop, n_mels=64):
    """mel-spectrogram 64 band -> MFCC 13 -> statistik (mean/std) MFCC & delta-MFCC = 52 dimensi."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N, hop_length=hop, win_length=N, window=win, n_mels=n_mels)
    L = librosa.power_to_db(S)
    mf = librosa.feature.mfcc(S=L, n_mfcc=13)
    d = librosa.feature.delta(mf, width=5, mode="nearest")
    return np.concatenate([mf.mean(1), mf.std(1), d.mean(1), d.std(1)])


def clip_all(path, sr, a, max_dur, specs, kinds, sos, fir, want_snr):
    """Mode streaming: SATU klip dimuat, diproses, difitur-kan untuk semua konfigurasi, lalu dibuang dari memori."""
    y = load_preemph(path, sr, a, max_dur)
    out, snr = {}, None
    for kind in kinds:
        yf = apply_filter(y, kind, sos, fir)
        for key, (N, w, hop) in specs.items():
            out[(kind, key)] = clip_features(yf, sr, N, w, hop)
        if want_snr and kind == "iir":
            snr = (snr_db(y), snr_db(yf))
    return out, snr


def stream(cfg, ctx, specs, kinds=("iir",), want_snr=False, desc="", chunk=200):
    """Ekstraksi fitur seluruh klip tanpa menyimpan sinyal. Mengembalikan ({(kind,key): X}, snr_array atau None)."""
    paths = list(ctx["df"].path); res, t0 = [], time.time()
    for i in range(0, len(paths), chunk):
        res += parallel(cfg, (delayed(clip_all)(p, cfg.sr, cfg.preemph, cfg.max_dur, specs, kinds, ctx["sos"], ctx["fir"], want_snr)
                              for p in paths[i:i+chunk]), 16)
        print(f"  {desc}: {min(i+chunk, len(paths))}/{len(paths)} klip ({time.time()-t0:.0f} s)", flush=True)
    feats = {k: np.vstack([r[0][k] for r in res]) for k in res[0][0]}
    snr = np.array([r[1] for r in res]) if want_snr else None
    return feats, snr


def load_filtered(cfg, ctx, i, kind="iir"):
    return apply_filter(load_preemph(ctx["df"].path.iloc[i], cfg.sr, cfg.preemph, cfg.max_dur), kind, ctx["sos"], ctx["fir"])


def B_preprocess(cfg, ctx):
    title("B. PRAPROSES")
    sos, fir = make_filters(cfg)
    # Respons frekuensi, fase, group delay: Butterworth orde 4 vs FIR (single-pass)
    w1, h1 = signal.sosfreqz(sos, worN=4096, fs=cfg.sr)
    w2, h2 = signal.freqz(fir, 1, worN=4096, fs=cfg.sr)
    b, a = signal.sos2tf(sos)
    wg1, gd1 = signal.group_delay((b, a), w=4096, fs=cfg.sr)
    wg2, gd2 = signal.group_delay((fir, [1.0]), w=4096, fs=cfg.sr)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].plot(w1, 20*np.log10(abs(h1)+1e-12), label="Butterworth orde 4")
    ax[0].plot(w2, 20*np.log10(abs(h2)+1e-12), label=f"FIR {cfg.fir_taps} tap")
    ax[0].set(ylim=(-80, 5), xlabel="Hz", ylabel="dB", title="Respons magnitudo"); ax[0].legend()
    ax[1].plot(w1, np.unwrap(np.angle(h1)), label="IIR"); ax[1].plot(w2, np.unwrap(np.angle(h2)), label="FIR")
    ax[1].set(xlabel="Hz", ylabel="rad", title="Respons fase"); ax[1].legend()
    ax[2].plot(wg1, gd1/cfg.sr*1000, label="IIR"); ax[2].plot(wg2, gd2/cfg.sr*1000, label="FIR")
    ax[2].set(xlim=(0, cfg.sr/2), xlabel="Hz", ylabel="ms", title="Group delay"); ax[2].legend()
    savefig(cfg, "B1_respons_filter.png")
    ctx.update(sos=sos, fir=fir)
    print("Mode streaming: sinyal tidak disimpan di memori (satu klip diproses lalu dibuang).")


# =============================================================================
# C. EKSPERIMEN INTI
# =============================================================================
def window_metrics(name, N=1024, nfft=2**16):
    w = signal.get_window(name, N)
    mag = np.abs(np.fft.rfft(w, nfft)); db = 20*np.log10(mag/mag.max() + 1e-12)
    k = np.arange(len(db)) * N / nfft                                    # sumbu dalam bin
    mn = np.where((db[1:-1] < db[:-2]) & (db[1:-1] < db[2:]))[0][0] + 1  # null pertama
    i3 = np.argmax(db < -3)
    return ({"window": NICE[name], "Peak sidelobe (dB)": db[mn:].max(),
             "Main lobe null-ke-null (bin)": 2*k[mn], "Lebar -3 dB (bin)": 2*k[i3]}, k, db)


def C_core(cfg, ctx):
    title("C. EKSPERIMEN INTI")
    df = ctx["df"]

    # C1 trade-off resolusi
    res = pd.DataFrame([{"N": N, "df (Hz) = fs/N": cfg.sr/N, "dt (ms) = N/fs": 1000*N/cfg.sr,
                         "hop (ms)": 1000*N*cfg.hop_ratio/cfg.sr} for N in WIN_SIZES]).round(2)
    print("\nC1. Resolusi waktu-frekuensi\n" + res.to_string(index=False))
    save_table(cfg, res, "C1_tabel_resolusi.csv", index=False)

    # C2 spektrogram satu klip burung
    cand = np.where(df.label == "VillageWeaver")[0]
    i = int(cand[0]) if len(cand) else int(np.where(df.label.isin(ctx["bird"]))[0][0])
    y = load_filtered(cfg, ctx, i)
    fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
    for a, N in zip(ax[0], WIN_SIZES):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=N, hop_length=N//2, window="hann")), ref=np.max)
        librosa.display.specshow(D, sr=cfg.sr, hop_length=N//2, x_axis="time", y_axis="hz", ax=a, vmin=-80)
        a.set_title(f"Hann N={N} (df={cfg.sr/N:.0f} Hz, dt={1000*N/cfg.sr:.0f} ms)")
    for a, w in zip(ax[1], WIN_TYPES):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=1024, hop_length=512, window=w)), ref=np.max)
        librosa.display.specshow(D, sr=cfg.sr, hop_length=512, x_axis="time", y_axis="hz", ax=a, vmin=-80)
        a.set_title(f"{NICE[w]} N=1024")
    fig.suptitle(f"Klip: {df.label[i]}"); savefig(cfg, "C2_spektrogram.png")

    # C3 leakage
    tab = pd.DataFrame([window_metrics(w)[0] for w in WIN_TYPES]).round(2)
    print("\nC3. Leakage tiap window (N=1024)\n" + tab.to_string(index=False))
    save_table(cfg, tab, "C3_tabel_window.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
    for w in WIN_TYPES:
        _, k, db = window_metrics(w); ax[0].plot(k, db, label=NICE[w])
    ax[0].set(xlim=(0, 12), ylim=(-120, 3), xlabel="Frekuensi (bin)", ylabel="dB",
              title="Respons spektral window (N=1024)"); ax[0].legend()
    N = 1024; f0 = 100.5 * cfg.sr / N; tone = np.sin(2*np.pi*f0*np.arange(N)/cfg.sr)   # di antara dua bin
    for w in WIN_TYPES:
        s_ = np.abs(np.fft.rfft(tone*signal.get_window(w, N))); s_ = 20*np.log10(s_/s_.max()+1e-12)
        ax[1].plot(np.arange(len(s_))*cfg.sr/N, s_, label=NICE[w])
    ax[1].set(xlim=(0, 6000), ylim=(-120, 3), xlabel="Hz", ylabel="dB", title=f"Leakage nada {f0:.0f} Hz"); ax[1].legend()
    savefig(cfg, "C3_leakage.png")

    # C4 biaya komputasi (100 klip contoh, tersebar merata)
    idx = np.linspace(0, len(df)-1, min(100, len(df))).astype(int)
    clips = [load_filtered(cfg, ctx, int(j)) for j in idx]; rows = []
    for N in WIN_SIZES:
        for w in WIN_TYPES:
            hop = int(N*cfg.hop_ratio); t = time.perf_counter()
            for c in clips: clip_features(c, cfg.sr, N, w, hop)
            ms = 1000*(time.perf_counter()-t)/len(clips); fr = 1 + len(clips[0])//hop
            rows.append({"window": NICE[w], "N": N, "ms_per_klip": ms, "frame": fr,
                         "ukuran_melspec(64xframe)": 64*fr, "dim_fitur_SVM": 52})
    del clips
    cost = pd.DataFrame(rows).round(2)
    print("\nC4. Waktu ekstraksi (ms/klip)\n" + cost.pivot(index="window", columns="N", values="ms_per_klip").to_string())
    save_table(cfg, cost, "C4_biaya_komputasi.csv", index=False)

    # C5 ekstraksi fitur 16 konfigurasi (+ baseline librosa) + SNR, satu kali lewat data
    specs = {f"{w}_{N}": (N, w, int(N*cfg.hop_ratio)) for N in WIN_SIZES for w in WIN_TYPES}
    specs["baseline_hann_2048_hop512"] = (2048, "hann", 512)
    f = cfg.out / f"fitur_{cfg.sub_train}_{cfg.sub_test}_{int(cfg.merge6)}_{cfg.max_dur:g}_utama.npz"
    if f.exists():
        z = np.load(f); feats = {k: z[k] for k in specs}; snr = z["snr"]
        print(f"  fitur dimuat dari cache: {f.name}")
    else:
        fe, snr = stream(cfg, ctx, specs, ("iir",), want_snr=True, desc="ekstraksi 17 konfigurasi")
        feats = {k: fe[("iir", k)] for k in specs}
        np.savez(f, snr=snr, **feats)
    ctx.update(feats=feats, snr=snr)


# =============================================================================
# D. VALIDASI DAN UJI STATISTIK
# =============================================================================
def make_svm(seed):
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                               class_weight="balanced", random_state=seed))


def run_fold(X, y, tr, va, seed):
    clf = make_svm(seed).fit(X[tr], y[tr]); p = clf.predict(X[va])
    return f1_score(y[va], p, average="macro"), accuracy_score(y[va], p)


def D_validation(cfg, ctx):
    title("D. VALIDASI & UJI STATISTIK")
    feats, y, is_tr, classes = ctx["feats"], ctx["y"], ctx["is_tr"], ctx["classes"]
    idx_tr, idx_te = np.where(is_tr)[0], np.where(~is_tr)[0]
    ytr, yte = y[idx_tr], y[idx_te]
    grid = [f"{w}_{N}" for N in WIN_SIZES for w in WIN_TYPES]
    folds = ctx["folds"]            # sudah disusun di A_data -> make_folds (pembagian sama persis)

    jobs = [(k, s, f, tr, va) for k in grid for s in cfg.seeds for f, (tr, va) in enumerate(folds[s])]
    t0 = time.time()
    out = parallel(cfg, (delayed(run_fold)(feats[k][idx_tr], ytr, tr, va, s) for k, s, f, tr, va in jobs))
    cv = pd.DataFrame([dict(key=k, seed=s, fold=f, macro_f1=o[0], acc=o[1]) for (k, s, f, _, _), o in zip(jobs, out)])
    cv["N"] = cv.key.str.split("_").str[1].astype(int); cv["window"] = cv.key.str.split("_").str[0]
    print(f"{len(jobs)} fit SVM selesai dalam {(time.time()-t0)/60:.1f} menit")
    save_table(cfg, cv, "D1_cv_hasil.csv", index=False)
    summ = cv.groupby("key")[["macro_f1", "acc"]].agg(["mean", "std"]).round(4).sort_values(("macro_f1", "mean"), ascending=False)
    print("\nD1. Hasil CV per konfigurasi\n" + summ.to_string())

    hm = cv.groupby(["window", "N"]).macro_f1.mean().unstack().loc[WIN_TYPES]; hm.index = [NICE[w] for w in hm.index]
    plt.figure(figsize=(6.5, 3.8)); sns.heatmap(hm, annot=True, fmt=".3f", cmap="YlGnBu"); plt.title("Rerata macro-F1 (CV)")
    savefig(cfg, "D1_heatmap_f1.png")

    # Friedman (blok = seed x fold) + Nemenyi
    M = cv.pivot_table(index=["seed", "fold"], columns="key", values="macro_f1")[grid]
    stat, p = friedmanchisquare(*[M[c] for c in M.columns])
    print(f"\nFriedman: chi2 = {stat:.2f}, p = {p:.3g} (blok={len(M)}, konfigurasi={M.shape[1]})")
    ranks = M.rank(axis=1, ascending=False).mean().sort_values()
    print("Rerata peringkat (1 = terbaik):\n" + ranks.round(2).to_string())
    nem = sp.posthoc_nemenyi_friedman(M.values); nem.index = nem.columns = M.columns
    save_table(cfg, nem, "D2_nemenyi_p.csv")
    order = ranks.index.tolist()
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(nem.loc[order, order], cmap="viridis_r", vmax=0.1, cbar_kws={"label": "p Nemenyi (>0,1 dipotong)"})
    plt.title("Post-hoc Nemenyi (gelap = beda signifikan)"); savefig(cfg, "D2_nemenyi.png")
    best = ranks.index[0]; print("Konfigurasi terbaik (peringkat):", best)
    save_table(cfg, ranks.rename("rerata_peringkat").to_frame(), "D2_peringkat.csv")

    # H1 & H2: Wilcoxon berpasangan per blok
    blk = lambda keys: M[keys].mean(axis=1)
    tests = [("H1: window 512-1024 vs 256 & 2048",
              blk([f"{w}_{N}" for w in WIN_TYPES for N in (512, 1024)]), blk([f"{w}_{N}" for w in WIN_TYPES for N in (256, 2048)])),
             ("H2: Hann/Blackman vs Rectangular",
              blk([f"{w}_{N}" for w in ("hann", "blackman") for N in WIN_SIZES]), blk([f"boxcar_{N}" for N in WIN_SIZES]))]
    h_rows = []
    for nm, a, b in tests:
        try:
            pv = wilcoxon(a, b).pvalue
        except ValueError:
            pv = np.nan
        h_rows.append(dict(uji=nm, selisih_macroF1=(a-b).mean(), p_wilcoxon=pv))
        print(f"{nm}: selisih rerata macro-F1 = {(a-b).mean():+.4f}, Wilcoxon p = {pv:.3g}")
    save_table(cfg, pd.DataFrame(h_rows), "D2_H1_H2.csv", index=False)
    print("Catatan: 3 seed memakai data yang sama -> blok tidak sepenuhnya independen; p cenderung optimis.")

    # Evaluasi akhir di data testing
    summary = []
    for key in [best, "baseline_hann_2048_hop512"]:
        clf = make_svm(cfg.seed).fit(feats[key][idx_tr], ytr); p_ = clf.predict(feats[key][idx_te])
        summary.append(dict(konfigurasi=key, macro_f1=f1_score(yte, p_, average="macro"), akurasi=accuracy_score(yte, p_)))
        if key == best:
            p_best = p_
    sdf = pd.DataFrame(summary).round(4)
    print("\nD3. Evaluasi akhir (data testing)\n" + sdf.to_string(index=False))
    save_table(cfg, sdf, "D3_test_akhir.csv", index=False)
    print("\n" + classification_report(yte, p_best, target_names=classes, digits=3))
    cm = confusion_matrix(yte, p_best, normalize="true")
    plt.figure(figsize=(8, 6.5)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Prediksi"); plt.ylabel("Sebenarnya"); plt.title(f"Matriks konfusi (test) - {best}"); savefig(cfg, "D3_konfusi.png")
    ctx.update(best=best, folds=folds, idx_tr=idx_tr, ytr=ytr)


# =============================================================================
# E. ANALISIS DAN PEMBAHASAN
# =============================================================================
def E_analysis(cfg, ctx):
    title("E. ANALISIS & PEMBAHASAN")
    lab = ctx["df"].label.values

    # E1: H3, SNR sebelum/sesudah bandpass (dihitung saat ekstraksi fitur)
    snr = pd.DataFrame({"label": lab, "sebelum": ctx["snr"][:, 0], "sesudah": ctx["snr"][:, 1]})
    snr["kelompok"] = np.where(snr.label == "Noise", "Noise", "Burung")
    b = snr[snr.kelompok == "Burung"]
    pv = wilcoxon(b.sesudah, b.sebelum).pvalue
    print(f"E1. Burung: SNR {b.sebelum.mean():.2f} dB -> {b.sesudah.mean():.2f} dB "
          f"(selisih {(b.sesudah-b.sebelum).mean():+.2f} dB), Wilcoxon p = {pv:.2e}, n = {len(b)}")
    yb = (snr.kelompok == "Burung").astype(int)
    print(f"    AUC burung-vs-noise dari SNR: {roc_auc_score(yb, snr.sebelum):.3f} -> {roc_auc_score(yb, snr.sesudah):.3f}"
          " (indikator kasar)")
    print(snr.groupby("label")[["sebelum", "sesudah"]].mean().round(2).to_string())
    save_table(cfg, snr, "E1_snr.csv", index=False)
    m = snr.melt(id_vars="kelompok", value_vars=["sebelum", "sesudah"], var_name="filter", value_name="SNR (dB)")
    plt.figure(figsize=(6, 4)); sns.boxplot(data=m, x="kelompok", y="SNR (dB)", hue="filter")
    plt.title("SNR sebelum vs sesudah bandpass"); savefig(cfg, "E1_snr.png")

    # E2 (tanpa filter / FIR) & E3 (hop 75%) untuk konfigurasi terbaik: satu lewat data tambahan
    best = ctx["best"]; wb, Nb = best.split("_")[0], int(best.split("_")[1])
    specs = {"best": (Nb, wb, int(Nb*cfg.hop_ratio)), "hop75": (Nb, wb, int(Nb*0.25))}
    f = cfg.out / f"fitur_{cfg.sub_train}_{cfg.sub_test}_{int(cfg.merge6)}_{cfg.max_dur:g}_E_{best}.npz"
    if f.exists():
        z = np.load(f); ex = {k: z[k] for k in z.files}
    else:
        fe, _ = stream(cfg, ctx, specs, ("none", "iir", "fir"), desc=f"E2/E3 ({best})")
        ex = {f"{k[0]}|{k[1]}": v for k, v in fe.items()}; np.savez(f, **ex)

    def cv_score(X):
        sc = [run_fold(X[ctx["idx_tr"]], ctx["ytr"], tr, va, 0)[0] for tr, va in ctx["folds"][cfg.seeds[0]]]
        return np.mean(sc), np.std(sc)
    rows = [dict(filter=k, macro_f1=cv_score(ex[f"{k}|best"])[0], std=cv_score(ex[f"{k}|best"])[1]) for k in ("none", "iir", "fir")]
    f2 = pd.DataFrame(rows).round(4)
    print(f"\nE2. Pengaruh filter ({best}, CV seed {cfg.seeds[0]})\n" + f2.to_string(index=False))
    save_table(cfg, f2, "E2_filter_vs_akurasi.csv", index=False)

    rows = []
    for name, key in (("50%", "iir|best"), ("75%", "iir|hop75")):
        mu, sd = cv_score(ex[key]); rows.append({"overlap": name, "macro_f1": mu, "std": sd})
    f3 = pd.DataFrame(rows).round(4)
    print("\nE3. Ablasi overlap/hop\n" + f3.to_string(index=False))
    save_table(cfg, f3, "E3_ablasi_hop.csv", index=False)


# =============================================================================
# F. LAPORAN DAN REPRODUKSIBILITAS
# =============================================================================
def F_report(cfg, ctx):
    title("F. REPRODUKSIBILITAS")
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               librosa=librosa.__version__, sklearn=sklearn.__version__, target_sr=cfg.sr, preemph=cfg.preemph,
               bandpass=[cfg.bp_low, cfg.bp_high], max_dur=cfg.max_dur, jobs=cfg.jobs, fir_taps=cfg.fir_taps, win_sizes=WIN_SIZES, win_types=WIN_TYPES,
               hop_ratio=cfg.hop_ratio, folds=cfg.folds, seeds=cfg.seeds, global_seed=cfg.seed,
               sub_train=cfg.sub_train, sub_test=cfg.sub_test, merge6=cfg.merge6, classes=ctx["classes"],
               n_train=int(ctx["is_tr"].sum()), n_test=int((~ctx["is_tr"]).sum()), best_config=ctx["best"],
               svm=dict(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))
    (cfg.out / "F_konfigurasi_eksperimen.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(env, indent=2, ensure_ascii=False))
    print(f"\nSemua luaran ada di: {cfg.out}")
    print("Keterbatasan untuk laporan: spesies Afrika Timur; label dari BirdNET (mungkin salah label); klip dari "
          "rekaman yang sama bisa muncul di fold training & validasi (CV cenderung optimis, karena itu ada test "
          "terpisah); kelas Noise bergaya ESC-50, belum tentu persis noise sawah.")


def main():
    cfg = get_args()
    np.random.seed(cfg.seed)
    t0 = time.time()
    ctx = A_data(cfg)
    B_preprocess(cfg, ctx)
    C_core(cfg, ctx)
    D_validation(cfg, ctx)
    E_analysis(cfg, ctx)
    F_report(cfg, ctx)
    print(f"\nSELESAI dalam {(time.time()-t0)/60:.1f} menit")


if __name__ == "__main__":
    main()
