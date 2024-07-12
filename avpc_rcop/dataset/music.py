"""
Mixed music PyTorch dataset.
"""

import os
import random
import numpy as np
from .base import BaseDataset


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix_train_pt = opt.num_mix_train_pt
        self.num_mix_train_ft = opt.num_mix_train_ft
        self.num_mix_eval = opt.num_mix_eval

    def __getitem__(self, index):

        if self.process_stage == 'train_pt':   # pre-training stage: A+B==>A, A+C==>A
            N = self.num_mix_train_pt   # default: 3
        elif self.process_stage == 'train_ft':   # fine-tuning stage: A+B+...==>A,B,...
            N = self.num_mix_train_ft   # default: 2
        else:   # evaluation or test stage: A+B==>A,B
            N = self.num_mix_eval   # default: 2

        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        sound_lib = []
        clip_time = np.zeros((N, 2))

        # the first video
        infos[0] = self.list_sample[index]
        sound_lib.append(infos[0][0].split('/')[-1])   # YouTubeID.wav

        # sample other videos for synthesizing mixture
        if self.process_stage == 'val' or self.process_stage == 'test':
            random.seed(index)
        for n in range(1, N):
            loop_flag = True
            while loop_flag:
                indexN = random.randint(0, len(self.list_sample) - 1)
                sound_ytid = self.list_sample[indexN][0].split('/')[-1]
                if sound_ytid not in sound_lib:
                    sound_lib.append(sound_ytid)
                    break
            infos[n] = self.list_sample[indexN]

        # select audio clips and video frames
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_chunksN, count_framesN = infoN

            if self.process_stage == 'val' or self.process_stage == 'test':
                random.seed(index + n)

            # audio clips
            # random, not to sample last chunks (discard uncorrelated ending)
            idx_chunk = random.randint(0, int(count_chunksN) - 2)
            # random (each chunk 20s)
            if idx_chunk == 0:  # start time >= 10s to discard salient beginning
                start_clip_sec = random.randint(10, 20 - 6 - 1)
            else:
                start_clip_sec = random.randint(0, 20 - 6 - 1)

            clip_time[n, 0] = idx_chunk
            clip_time[n, 1] = start_clip_sec

            path_audio = os.path.join(path_audioN, '{:06d}.wav'.format(idx_chunk))
            audios[n] = self._load_audio(path_audio, start_clip_sec)
            audios[n] = self.normalize(audios[n])

            # video frames
            center_timeN = idx_chunk * 20 + start_clip_sec + self.audSec / 2
            center_frameN = int(center_timeN * self.frameRate)
            path_frames = []
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames.append(
                    os.path.join(
                        path_frameN, '{:06d}.jpg'.format(center_frameN + idx_offset)))

            frames[n] = self._load_frames(path_frames)

        # transform audio with STFT
        if self.process_stage == 'train_pt':
            mag_mix, mags, _ = self._mix_n_and_stft(audios, mult_mix=True)
            ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        elif self.process_stage == 'train_ft':
            mag_mix, mags, _ = self._mix_n_and_stft(audios)
            ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        else:
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)
            ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags,
                        'audios': audios, 'phase_mix': phase_mix, 'infos': infos, 'clip_time': clip_time}

        return ret_dict

    def normalize(self, audio_data, re_factor=0.8):
        EPS = 1e-3
        min_data = audio_data.min()
        audio_data -= min_data
        max_data = audio_data.max()
        audio_data /= max_data + EPS
        audio_data -= 0.5
        audio_data *= 2
        if self.process_stage == 'train_pt' or self.process_stage == 'train_ft':
            re_factor = random.random() + 0.5   # 0.5-1.5
            audio_data *= re_factor
        return audio_data.clip(-1, 1)
