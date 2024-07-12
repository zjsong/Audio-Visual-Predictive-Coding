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
        self.num_mix = opt.num_mix

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        clip_time = np.zeros((N, 2))

        # the first video
        infos[0] = self.list_sample[index]
        # instrument1_name = infos[0][0].split('/')[-2]
        sound1_ytid = infos[0][0].split('/')[-1]   # YouTubeID.wav

        # sample other videos for synthesizing mixture
        if not self.process_stage == 'train':
            random.seed(index)
        for n in range(1, N):
            loop_flag = True
            while loop_flag:
                indexN = random.randint(0, len(self.list_sample) - 1)
                # instrument2_name = self.list_sample[indexN][0].split('/')[-2]
                sound2_ytid = self.list_sample[indexN][0].split('/')[-1]
                # if indexN != index:
                # if instrument2_name != instrument1_name:
                if sound2_ytid != sound1_ytid:
                    break

            infos[n] = self.list_sample[indexN]

        # select audio clips and video frames
        for n, infoN in enumerate(infos):
            path_audioN, path_frameN, count_chunksN, count_framesN = infoN

            if not self.process_stage == 'train':
                random.seed(index + n)
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

            # video frames (extracted with the given stride)
            center_timeN = idx_chunk * 20 + start_clip_sec + self.audSec / 2
            center_frameN = int(center_timeN * self.frameRate)
            path_frames = []
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames.append(
                    os.path.join(
                        path_frameN, '{:06d}.jpg'.format(center_frameN + idx_offset)))

            frames[n] = self._load_frames(path_frames)

        mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        ret_dict = {'mag_mix': mag_mix, 'frames': frames, 'mags': mags}
        if self.process_stage != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos
            ret_dict['clip_time'] = clip_time

        return ret_dict

    def normalize(self, audio_data, re_factor=0.8):
        EPS = 1e-3
        min_data = audio_data.min()
        audio_data -= min_data
        max_data = audio_data.max()
        audio_data /= max_data + EPS
        audio_data -= 0.5
        audio_data *= 2
        if self.process_stage == 'train':
            re_factor = random.random() + 0.5   # 0.5-1.5
            audio_data *= re_factor
        return audio_data.clip(-1, 1)
