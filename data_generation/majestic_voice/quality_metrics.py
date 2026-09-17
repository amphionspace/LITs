"""Resident speaker/DNSMOS models; weights are loaded only from local assets.

DNSMOS windowing and calibration follow microsoft/DNS-Challenge/DNSMOS/dnsmos_local.py.
Speaker architectures come from the existing UltraEval-Audio installation.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

ASSETS = Path(os.environ.get('MAJESTIC_VOICE_ASSETS_ROOT', '/119010446/tts-assets'))
ULTRA = Path(os.environ.get('ULTRAEVAL_AUDIO_ROOT', '/119010446/UltraEval-Audio'))


def audio16(path):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f'Invalid mono audio: {path}')
    if sr != 16000:
        divisor = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // divisor, sr // divisor)
    return np.asarray(audio, dtype=np.float32)


class DNSMOS:
    def __init__(self, provider='cpu', device_id=0):
        import onnxruntime as ort
        if provider not in ('cpu','cuda'):raise ValueError('DNSMOS provider must be cpu or cuda')
        providers=['CPUExecutionProvider']
        if provider=='cuda':
            if 'CUDAExecutionProvider' not in ort.get_available_providers():raise RuntimeError('ONNX Runtime CUDA provider is unavailable')
            if hasattr(ort,'preload_dlls'):ort.preload_dlls()
            providers=[('CUDAExecutionProvider',{'device_id':device_id,'cudnn_conv_algo_search':'HEURISTIC','use_tf32':0}),'CPUExecutionProvider']
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        root = ASSETS / 'DNSMOS' / 'DNSMOS'
        self.main = ort.InferenceSession(str(root / 'sig_bak_ovr.onnx'), opts,
                                         providers=providers)
        self.p808 = ort.InferenceSession(str(root / 'model_v8.onnx'), opts,
                                         providers=providers)
        if provider=='cuda' and any(s.get_providers()[0]!='CUDAExecutionProvider' for s in (self.main,self.p808)):
            raise RuntimeError('DNSMOS unexpectedly fell back to CPU')
        self.provider=provider

    def __call__(self, audio):
        import librosa
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            raise ValueError('Empty audio')
        window = 144160  # 9.01 s at 16 kHz
        while len(audio) < window:
            audio = np.concatenate([audio, audio])
        scores = []
        # Preserve the reference implementation's window-count formula exactly.
        num_hops = int(np.floor(len(audio) / 16000) - 9.01) + 1
        for index in range(num_hops):
            start = index * 16000
            segment = audio[start:start + window]
            raw = self.main.run(None, {'input_1': segment[None]})[0][0]
            mel = librosa.feature.melspectrogram(y=segment[:-160], sr=16000,
                                                n_fft=321, hop_length=160, n_mels=120)
            mel = ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T
            p808 = float(self.p808.run(None, {'input_1': mel.astype('float32')[None]})[0][0][0])
            sig, bak, ovr = raw
            scores.append([
                np.polyval([-0.08397278, 1.22083953, 0.0052439], sig),
                np.polyval([-0.13166888, 1.60915514, -0.39604546], bak),
                np.polyval([-0.06766283, 1.11546468, 0.04602535], ovr), p808,
            ])
        return {**dict(zip(('dnsmos_sig', 'dnsmos_bak', 'dnsmos_ovrl', 'dnsmos_p808'),
                           map(float, np.mean(scores, axis=0)))),
                'dnsmos_protocol': 'dns-challenge-window-count-v1'}


class SpeakerMetrics:
    def __init__(self, reference, device='cuda:0', fast_matmul=False,
                 admission_thresholds=(0.60, 0.60)):
        import torchaudio
        self.torchaudio = torchaudio
        self.device = device
        self.fast_matmul = fast_matmul
        self.admission_thresholds = admission_thresholds
        if fast_matmul:
            torch.set_float32_matmul_precision('highest')
        sys.path.insert(0, str(ULTRA / 'audio_evals/lib/simo'))
        sys.path.insert(0, str(ULTRA / 'audio_evals/lib/cv3_speaker_sim/3D-Speaker'))
        os.environ['WAVLM_LARGE_BACKBONE'] = str(ASSETS / 'WavLM-Large/wavlm_large.pt')
        from models_ecapa_tdnn import ECAPA_TDNN_SMALL
        from speakerlab.models.campplus.DTDNN import CAMPPlus
        self.wavlm = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
        checkpoint = torch.load(ASSETS / 'WavLM-Large/wavlm_large_finetune.pth',
                                map_location='cpu', weights_only=False)
        result = self.wavlm.load_state_dict(checkpoint['model'], strict=False)
        # Training-only speaker classification head is absent from the embedding model.
        unexpected = set(result.unexpected_keys) - {'loss_calculator.projection.weight'}
        if result.missing_keys or unexpected:
            raise RuntimeError(f'WavLM checkpoint mismatch: {result}')
        self.wavlm = self.wavlm.to(device).eval()
        self.camp = CAMPPlus(feat_dim=80, embedding_size=192)
        checkpoint = torch.load(ASSETS / 'CAMPlus/campplus_cn_common.bin',
                                map_location='cpu', weights_only=False)
        self.camp.load_state_dict(checkpoint, strict=True)
        self.camp = self.camp.to(device).eval()
        self.reference = self.embeddings(audio16(reference))

    @torch.inference_mode()
    def embeddings(self, audio):
        waveform = torch.from_numpy(audio).unsqueeze(0)
        wavlm = F.normalize(self.wavlm(waveform.to(self.device)), dim=-1)
        features = self.torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=80, sample_frequency=16000, dither=0)
        features = features - features.mean(dim=0, keepdim=True)
        camp = F.normalize(self.camp(features.unsqueeze(0).to(self.device)), dim=-1)
        return wavlm, camp

    def __call__(self, audio):
        wavlm, camp = self.embeddings(audio)
        return {'wavlm_similarity': float((wavlm * self.reference[0]).sum()),
                'camp_similarity': float((camp * self.reference[1]).sum())}

    @torch.inference_mode()
    def batch(self, audios, max_batch=8):
        # Batch only equal-length waveforms: no padding can alter speaker scores.
        if self.fast_matmul:
            torch.set_float32_matmul_precision('high')
        groups = {}
        for index, audio in enumerate(audios):
            groups.setdefault(len(audio), []).append(index)
        results = [None] * len(audios)
        for indices in groups.values():
            for offset in range(0, len(indices), max_batch):
                group = indices[offset:offset + max_batch]
                waveforms = torch.from_numpy(np.stack([audios[i] for i in group]))
                wavlm = F.normalize(self.wavlm(waveforms.to(self.device)), dim=-1)
                features = []
                for waveform in waveforms:
                    fbank = self.torchaudio.compliance.kaldi.fbank(
                        waveform[None], num_mel_bins=80, sample_frequency=16000, dither=0)
                    features.append(fbank - fbank.mean(dim=0, keepdim=True))
                camp = F.normalize(self.camp(torch.stack(features).to(self.device)), dim=-1)
                w = (wavlm * self.reference[0]).sum(dim=-1).cpu().tolist()
                c = (camp * self.reference[1]).sum(dim=-1).cpu().tolist()
                for index, wavlm_score, camp_score in zip(group, w, c):
                    results[index] = {'wavlm_similarity': wavlm_score, 'camp_similarity': camp_score}
        if self.fast_matmul:
            torch.set_float32_matmul_precision('highest')
            for index, result in enumerate(results):
                if any(abs(result[key] - threshold) < 0.002 for key, threshold in zip(
                    ('wavlm_similarity', 'camp_similarity'), self.admission_thresholds)):
                    results[index] = self(audios[index])
        return results
