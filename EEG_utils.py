import numpy as np
# import
from skimage.metrics import structural_similarity

def calculate_batch_ssim_new(img1, img2):
    ssim_value = structural_similarity(img1, img2, win_size=5, channel_axis=0,data_range=7)
    return ssim_value

def calculate_batch_psnr_new(img_batch1, img_batch2, max_value=1):
    assert img_batch1.shape == img_batch2.shape
    batch_size = img_batch1.shape[0]
    psnr_sum = 0.0
    for i in range(batch_size):
        mse = np.mean((img_batch1[i].astype(np.float64) - img_batch2[i].astype(np.float64)) ** 2)
        if mse == 0:
            psnr_sum += float('inf')
        else:
            psnr_sum += 20 * np.log10(max_value / np.sqrt(mse))
    
    if np.isinf(psnr_sum):
        return float('inf')
    
    avg_psnr = psnr_sum / batch_size
    return avg_psnr


def calculate_psnr(signal1, signal2, max_pixel_value=20):
    mse = np.mean((signal1 - signal2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_batch_psnr(original_signals, watermarked_signals):

    batch_size,  channels, samples = original_signals.shape
    
    psnr_results = []
    
    for batch_idx in range(batch_size):
        original_signal = original_signals[batch_idx, 0].flatten()
        watermarked_signal = watermarked_signals[batch_idx, 0].flatten()

        psnr = calculate_psnr(original_signal, watermarked_signal, max_pixel_value=200)
        psnr_results.append(psnr)

    avg_psnr = np.mean(psnr_results)
    
    return avg_psnr



import numpy as np

def calculate_ssim_for_1d(signal1, signal2, k1=0.01, k2=0.03, dynamic_range=4):

    assert len(signal1) == len(signal2), "Signals must have the same length."

    mu1 = np.mean(signal1)
    mu2 = np.mean(signal2)

    sigma1 = np.var(signal1)
    sigma2 = np.var(signal2)

    sigma12 = np.cov(signal1, signal2, bias=True)[0, 1]

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)

    ssim_value = numerator / denominator

    return ssim_value

def calculate_batch_ssim(original_signals, watermarked_signals):
    batch_size,channels, samples = original_signals.shape
    
    ssim_results = []
    
    for batch_idx in range(batch_size):

        original_signal = original_signals[batch_idx, 0].flatten()
        watermarked_signal = watermarked_signals[batch_idx, 0].flatten()
        
        ssim = calculate_ssim_for_1d(original_signal, watermarked_signal)
        ssim_results.append(ssim)
    
    avg_ssim = np.mean(ssim_results)    
    return avg_ssim


import numpy as np

def calculate_ncc(x, y):
    assert len(x) == len(y), "Signals must have the same length."

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    deviation_x = x - mean_x
    deviation_y = y - mean_y

    numerator = np.sum(deviation_x * deviation_y)

    denominator = np.sqrt(np.sum(deviation_x ** 2)) * np.sqrt(np.sum(deviation_y ** 2))

    if denominator == 0:
        return 0
    return numerator / denominator

def calculate_batch_ncc(original_signals, watermarked_signals):
    batch_size,  channels, samples = original_signals.shape
    
    ncc_results = []
    
    for batch_idx in range(batch_size):
        original_signal = original_signals[batch_idx, 0].flatten()
        watermarked_signal = watermarked_signals[batch_idx, 0].flatten()
        
        ncc = calculate_ncc(original_signal, watermarked_signal)
        ncc_results.append(ncc)

    avg_ncc = np.mean(ncc_results)
    
    return avg_ncc


import numpy as np

def calculate_kcd(original_signal, watermarked_signal):
    assert len(original_signal) == len(watermarked_signal), "Signals must have the same length."

    mean_original = np.mean(original_signal)
    mean_watermarked = np.mean(watermarked_signal)

    deviation_original = original_signal - mean_original
    deviation_watermarked = watermarked_signal - mean_watermarked

    numerator = np.sum(deviation_original * deviation_watermarked)

    denominator = np.sqrt(np.sum(deviation_original ** 2)) * np.sqrt(np.sum(deviation_watermarked ** 2))

    if denominator == 0:
        return 0

    return numerator / denominator

def calculate_batch_kcd(original_signals, watermarked_signals):

    batch_size,  channels, samples = original_signals.shape
    
    kcd_results = []
    
    for batch_idx in range(batch_size):
        original_signal = original_signals[batch_idx, 0].flatten()
        watermarked_signal = watermarked_signals[batch_idx, 0].flatten()
        
        kcd = calculate_kcd(original_signal, watermarked_signal)
        kcd_results.append(kcd)
    
    avg_kcd = np.mean(kcd_results)
    
    return avg_kcd



from scipy.stats import entropy

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def signal_to_distribution(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    bins = np.linspace(min_val, max_val, num=100)
    hist, _ = np.histogram(signal, bins=bins, density=True)
    return hist

def calculate_batch_jsd(original_signals, watermarked_signals):
    batch_size,  channels, samples = original_signals.shape
    
    jsd_results = []
    
    for batch_idx in range(batch_size):
        original_signal = original_signals[batch_idx, 0]
        watermarked_signal = watermarked_signals[batch_idx, 0]
        
        p = signal_to_distribution(original_signal)
        q = signal_to_distribution(watermarked_signal)
        
        jsd = jensen_shannon_divergence(p, q)
        jsd_results.append(jsd)
    
    avg_jsd = np.mean(jsd_results)
    
    return avg_jsd
