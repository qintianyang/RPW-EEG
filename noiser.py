
import torch.nn as nn
import torch
import numpy as np
from mne.filter import notch_filter
from torch.nn.functional import one_hot

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, noised_and_cover):
        return noised_and_cover
    
# 时域变化

# 增加高斯噪声
class add_gaussian_noise(nn.Module):
    def __init__(self, std=0.01):
        super(add_gaussian_noise, self).__init__()
        self.std = std
    def forward(self, noised_and_cover):
        noise = torch.randn_like(noised_and_cover) * self.std
        noised_and_cover_noise = noised_and_cover + noise
        return noised_and_cover_noise
    
# 平滑的变为0
class SmoothTimeMask(nn.Module):
    def __init__(self, mask_len_samples):
        super(SmoothTimeMask, self).__init__()
        self.mask_len_samples = mask_len_samples

    def forward(self, X):
        mask_start_per_sample = torch.tensor([10, 30])
        batch_size, n_channels, seq_len = X.shape
        t = torch.arange(seq_len, device=X.device).float().unsqueeze(0).unsqueeze(0)
        t = t.expand(batch_size, n_channels, seq_len)
        mask_start_per_sample = mask_start_per_sample.view(batch_size, 1, 1).expand(batch_size, n_channels, seq_len)
       
        s = 1000 / seq_len
        mask = (
            torch.sigmoid(s * -(t - mask_start_per_sample)) +
            torch.sigmoid(s * (t - mask_start_per_sample - self.mask_len_samples))
        ).float().to(X.device)
        return X * mask

# 时间轴的反转
class time_reverse(nn.Module):
    def __init__(self):
        super(time_reverse, self).__init__()
    def forward(self, noised_and_cover):
        noised_and_coverr = torch.flip(noised_and_cover, [-1])
        return noised_and_coverr

# 符号的反转
class sign_reverse(nn.Module):
    def __init__(self):
        super(sign_reverse, self).__init__()
    def forward(self, noised_and_cover):
        noised_and_coverr = -noised_and_cover
        return noised_and_coverr
        
# 频域的变化

# 频域的移位
class FrequencyShift(nn.Module):
    def __init__(self, shift_amount, dim=-1):
        super(FrequencyShift, self).__init__()
        self.shift_amount = shift_amount
        self.dim = dim

    def forward(self, X):
        # Perform Fourier transform
        X_fft = torch.fft.rfft(X, dim=self.dim)

        # Apply frequency shift
        if self.shift_amount > 0:
            X_fft = torch.cat((X_fft[:, :, self.shift_amount:], X_fft[:, :, :self.shift_amount]), dim=self.dim)
        elif self.shift_amount < 0:
            shift_amount = -self.shift_amount
            X_fft = torch.cat((X_fft[:, :, -shift_amount:], X_fft[:, :, :-shift_amount]), dim=self.dim)

        # Perform inverse Fourier transform
        X_shifted = torch.fft.irfft(X_fft, n=X.size(self.dim), dim=self.dim)

        return X_shifted
    
# 傅里叶变换
def check_random_state(seed):
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(f"Invalid random state: {seed}")
def _new_random_fft_phase_odd(batch_size, c, n, device, random_state):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random((batch_size, c, (n - 1) // 2))
    ).to(device)
    return torch.cat(
        [
            torch.zeros((batch_size, c, 1), device=device),
            random_phase,
            -torch.flip(random_phase, [-1]),
        ],
        dim=-1,
    )
def _new_random_fft_phase_even(batch_size, c, n, device, random_state):
    rng = check_random_state(random_state)
    random_phase = torch.from_numpy(
        2j * np.pi * rng.random((batch_size, c, n // 2 - 1))
    ).to(device)
    return torch.cat(
        [
            torch.zeros((batch_size, c, 1), device=device),
            random_phase,
            torch.zeros((batch_size, c, 1), device=device),
            -torch.flip(random_phase, [-1]),
        ],
        dim=-1,
    )
_new_random_fft_phase = {0: _new_random_fft_phase_even, 1: _new_random_fft_phase_odd}
class FTSurrogate(nn.Module):
    def __init__(self, phase_noise_magnitude, channel_indep, random_state=None):
        super(FTSurrogate, self).__init__()
        self.phase_noise_magnitude = phase_noise_magnitude
        self.channel_indep = channel_indep
        self.random_state = random_state

    def forward(self, X):
        assert (
            isinstance(self.phase_noise_magnitude, (float, torch.FloatTensor, torch.cuda.FloatTensor))
            and 0 <= self.phase_noise_magnitude <= 1
        ), f"phase_noise_magnitude must be a float between 0 and 1. Got {self.phase_noise_magnitude}."

        f = torch.fft.rfft(X.double(), dim=-1)
        device = X.device

        n = f.shape[-1]
        random_phase = _new_random_fft_phase[n % 2](
            f.shape[0],
            f.shape[-2] if self.channel_indep else 1,
            n,
            device=device,
            random_state=self.random_state,
        )
        if not self.channel_indep:
            random_phase = random_phase.expand(-1, f.shape[-2], -1)
        if isinstance(self.phase_noise_magnitude, torch.Tensor):
            self.phase_noise_magnitude = self.phase_noise_magnitude.to(device)
        f_shifted = f * torch.exp(self.phase_noise_magnitude * random_phase)
        shifted = torch.fft.irfft(f_shifted, n=X.shape[-1], dim=-1)
        transformed_X = shifted.real.float()

        return transformed_X


# 带通滤波器
class BandStopFilter(nn.Module):
    def __init__(self, sfreq, bandwidth, freqs_to_notch):
        super(BandStopFilter, self).__init__()
        self.sfreq = sfreq
        self.bandwidth = bandwidth
        self.freqs_to_notch = freqs_to_notch

    def forward(self, X):
        if self.bandwidth == 0:
            return X

        transformed_X = X.clone()
        for c, (sample, notched_freq) in enumerate(zip(transformed_X, self.freqs_to_notch)):
            # sample = sample.cpu().numpy().astype(np.float64)
            sample = sample.clone().cpu().detach().numpy().astype(np.float64)
            # sample = sample.cpu().detach().numpy().astype(np.float64)
            filtered_sample = notch_filter(
                sample,
                Fs=self.sfreq,
                freqs=notched_freq,
                method="fir",
                notch_widths=self.bandwidth,
                verbose=False,
            )
            transformed_X[c] = torch.as_tensor(filtered_sample, device=X.device)

        return transformed_X
    
    
    
# 空域上的变换

#通道对称
def check_random_state(seed):
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(f"Invalid random state: {seed}")

class ChannelSymmetry(nn.Module):
    def __init__(self, p_symmetry, pairs, random_state=None):
        super(ChannelSymmetry, self).__init__()
        self.p_symmetry = p_symmetry
        self.pairs = pairs
        self.random_state = random_state

    def forward(self, X):
        if self.p_symmetry == 0:
            return X
        rng = check_random_state(self.random_state)
        batch_size, n_channels, seq_len = X.shape

        # Create a copy of the input tensor to avoid modifying the original data
        transformed_X = X.clone()
        # Determine which pairs to swap
        swap_mask = torch.tensor(rng.binomial(1, self.p_symmetry, size=(batch_size, len(self.pairs))),
                                 dtype=torch.bool, device=X.device)
        for b in range(batch_size):
            for i, (c1, c2) in enumerate(self.pairs):
                if swap_mask[b, i]:
                    transformed_X[b, c1, :], transformed_X[b, c2, :] = transformed_X[b, c2, :], transformed_X[b, c1, :]

        return transformed_X

#通道丢弃
def check_random_state(seed):
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(f"Invalid random state: {seed}")

class ChannelDropout(nn.Module):
    def __init__(self, p_drop, random_state=None):
        super(ChannelDropout, self).__init__()
        self.p_drop = p_drop
        self.random_state = random_state

    def forward(self, X):
        if self.p_drop == 0:
            return X

        rng = check_random_state(self.random_state)
        batch_size, n_channels, seq_len = X.shape

        # Generate a mask to drop channels
        mask = torch.tensor(rng.binomial(1, 1 - self.p_drop, size=(batch_size, n_channels, 1)),
                            dtype=torch.float32, device=X.device)
        
        # Apply the mask to the input tensor
        transformed_X = X * mask

        return transformed_X

#通道切换
def check_random_state(seed):
    if seed is None or isinstance(seed, int):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError(f"Invalid random state: {seed}")

def _pick_channels_randomly(X, p_keep, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    mask = torch.tensor(rng.binomial(1, p_keep, size=(batch_size, n_channels)), dtype=torch.float32, device=X.device)
    return mask

def _make_permutation_matrix(X, mask, random_state):
    rng = check_random_state(random_state)
    batch_size, n_channels, _ = X.shape
    hard_mask = mask.round()
    batch_permutations = torch.empty(
        batch_size, n_channels, n_channels, device=X.device
    )
    for b, mask in enumerate(hard_mask):
        channels_to_shuffle = torch.arange(n_channels, device=X.device)
        channels_to_shuffle = channels_to_shuffle[mask.bool()]
        reordered_channels = torch.tensor(
            rng.permutation(channels_to_shuffle.cpu()), device=X.device
        )
        channels_permutation = torch.arange(n_channels, device=X.device)
        channels_permutation[channels_to_shuffle] = reordered_channels
        batch_permutations[b, ...] = one_hot(channels_permutation, num_classes=n_channels).float()
    return batch_permutations

class ChannelsShuffle(nn.Module):
    def __init__(self, p_shuffle, random_state=None):
        super(ChannelsShuffle, self).__init__()
        self.p_shuffle = p_shuffle
        self.random_state = random_state

    def forward(self, X):
        if self.p_shuffle == 0:
            return X
        mask = _pick_channels_randomly(X, 1 - self.p_shuffle, self.random_state)
        batch_permutations = _make_permutation_matrix(X, mask, self.random_state)
        transformed_X = torch.matmul(batch_permutations, X)
        return transformed_X
    
    
class Noiser(nn.Module):
    def __init__(self, device):
        super(Noiser, self).__init__()
        self.device = device
        self.Identity = Identity()
        self.add_gaussian_noise = add_gaussian_noise(std=0.01)
        self.SmoothTimeMask = SmoothTimeMask(mask_len_samples=100)
        self.time_reverse = time_reverse()
        self.sign_reverse = sign_reverse()
        self.spatial_noise = [self.time_reverse]
        # self.spatial_noise = [self.Identity,self.add_gaussian_noise,self.SmoothTimeMask]
        
        self.FrequencyShift = FrequencyShift(shift_amount=10)
        self.BandStopFilter = BandStopFilter(sfreq=500, bandwidth=10, freqs_to_notch=[50, 100, 150])
        self.FTSurrogate = FTSurrogate(phase_noise_magnitude=0.1, channel_indep=True, random_state=42)
        # self.frequency_noise = [self.FrequencyShift, self.BandStopFilter, self.FTSurrogate]
        self.frequency_noise = [self.FrequencyShift,  self.FTSurrogate]


        self.ChannelSymmetry = ChannelSymmetry(p_symmetry=0.5, pairs=[(0, 1), (2, 3)], random_state=42)
        self.ChannelDropout = ChannelDropout(p_drop=0.5, random_state=42)        
        self.ChannelsShuffle = ChannelsShuffle(p_shuffle=0.5, random_state=42)        
        self.temporal_noise = [self.ChannelSymmetry, self.ChannelDropout, self.ChannelsShuffle]
        
        noise_layers = [self.spatial_noise, self.frequency_noise, self.temporal_noise]

    def forward(self, x):
        # 从每个噪声类别中随机选择一个具体的噪声层
        x = torch.squeeze(x)
        spatial_noise_layer = np.random.choice(self.spatial_noise, 1)[0]
        x = spatial_noise_layer(x)
        # frequency_noise_layer = np.random.choice(self.frequency_noise, 1)[0]
        # x = frequency_noise_layer(x)
        # temporal_noise_layer = np.random.choice(self.temporal_noise, 1)[0]
        # x= temporal_noise_layer(x)
        x = x.to(self.device)
        x = torch.unsqueeze(x, 1)
        
        return x
