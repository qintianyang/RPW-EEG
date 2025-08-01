import numpy as np
# import
from skimage.metrics import structural_similarity

def calculate_batch_ssim_new(img1, img2):
    ssim_value = structural_similarity(img1, img2, win_size=5, channel_axis=0,data_range=1)
    return ssim_value

def calculate_batch_psnr_new(img_batch1, img_batch2, max_value=1):
    """
    计算两个图像批次的PSNR值，并返回平均PSNR值。
    
    参数:
        img_batch1 (numpy.ndarray): 第一批次的图像，形状为 (batchsize, H, W)
        img_batch2 (numpy.ndarray): 第二批次的图像，形状为 (batchsize, H, W)
        max_value (int): 图像的最大像素值，默认为255
    
    返回:
        float: 批次中所有图像对的平均PSNR值
    """
    assert img_batch1.shape == img_batch2.shape, "两个图像批次的形状必须相同"
    
    batch_size = img_batch1.shape[0]
    psnr_sum = 0.0
    
    for i in range(batch_size):
        mse = np.mean((img_batch1[i].astype(np.float64) - img_batch2[i].astype(np.float64)) ** 2)
        if mse == 0:
            psnr_sum += float('inf')
        else:
            psnr_sum += 20 * np.log10(max_value / np.sqrt(mse))
    
    # 如果有无穷大的值，直接返回无穷大
    if np.isinf(psnr_sum):
        return float('inf')
    
    # 否则计算平均PSNR值
    avg_psnr = psnr_sum / batch_size
    return avg_psnr




def calculate_psnr(signal1, signal2, max_pixel_value=20):
    """
    计算两个信号之间的峰值信噪比 (PSNR)。
    参数:
    signal1 : numpy.ndarray
        第一个信号，通常是原始信号。
    signal2 : numpy.ndarray
        第二个信号，通常是含有水印的信号。
    max_pixel_value : float
        信号中值的最大可能值，默认为 200。
    返回:
    float
        PSNR 值。
    """
    # 计算均方误差 (MSE)
    mse = np.mean((signal1 - signal2) ** 2)
    if mse == 0:
        # 如果 MSE 为 0，则表示两个信号完全相同
        return float('inf')
    # 根据 PSNR 公式计算 PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

def calculate_batch_psnr(original_signals, watermarked_signals):
    """
    计算原始信号和嵌入水印后信号之间的PSNR，并返回所有batch的平均值。
    """
    batch_size,  channels, samples = original_signals.shape
    
    psnr_results = []
    
    for batch_idx in range(batch_size):
        # 获取单个batch的数据
        original_signal = original_signals[batch_idx, 0].flatten()
        watermarked_signal = watermarked_signals[batch_idx, 0].flatten()
        
        # 计算PSNR
        psnr = calculate_psnr(original_signal, watermarked_signal, max_pixel_value=200)
        psnr_results.append(psnr)
    
    # 计算所有batch的平均PSNR
    avg_psnr = np.mean(psnr_results)
    
    return avg_psnr



import numpy as np

def calculate_ssim_for_1d(signal1, signal2, k1=0.01, k2=0.03, dynamic_range=4):
    """
    计算两个一维信号之间的 SSIM 值。

    参数:
    signal1 : numpy.ndarray
        第一个一维信号。
    signal2 : numpy.ndarray
        第二个一维信号。
    k1 : float
        SSIM 公式中的常数。
    k2 : float
        SSIM 公式中的常数。
    dynamic_range : float
        信号值的动态范围，默认为 1.0。

    返回:
    float
        SSIM 值。
    """

    # 确保信号长度相同
    assert len(signal1) == len(signal2), "Signals must have the same length."

    # 计算平均值
    mu1 = np.mean(signal1)
    mu2 = np.mean(signal2)

    # 计算方差
    sigma1 = np.var(signal1)
    sigma2 = np.var(signal2)

    # 计算协方差
    sigma12 = np.cov(signal1, signal2, bias=True)[0, 1]

    # 计算 SSIM
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2)

    ssim_value = numerator / denominator

    return ssim_value

def calculate_batch_ssim(original_signals, watermarked_signals):
    """
    计算原始信号和嵌入水印后信号之间的SSIM，并返回所有batch的平均值。
    """
    batch_size,channels, samples = original_signals.shape
    
    ssim_results = []
    
    for batch_idx in range(batch_size):
        # 获取单个batch的数据
        original_signal = original_signals[batch_idx, 0].flatten()
        watermarked_signal = watermarked_signals[batch_idx, 0].flatten()
        
        # 计算SSIM
        ssim = calculate_ssim_for_1d(original_signal, watermarked_signal)
        ssim_results.append(ssim)
    
    # 计算所有batch的平均SSIM
    avg_ssim = np.mean(ssim_results)    
    return avg_ssim


import numpy as np

def calculate_ncc(x, y):
    """
    计算两个一维信号之间的归一化互相关系数。
    """
    # 确保信号长度相同
    assert len(x) == len(y), "Signals must have the same length."

    # 计算均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 计算偏差
    deviation_x = x - mean_x
    deviation_y = y - mean_y

    # 计算分子
    numerator = np.sum(deviation_x * deviation_y)

    # 计算分母
    denominator = np.sqrt(np.sum(deviation_x ** 2)) * np.sqrt(np.sum(deviation_y ** 2))

    # 避免除以零
    if denominator == 0:
        return 0

    # 返回NCC值
    return numerator / denominator

def calculate_batch_ncc(original_signals, watermarked_signals):
    """
    计算原始信号和嵌入水印后信号之间的NCC，并返回所有batch的平均值。
    """
    batch_size,  channels, samples = original_signals.shape
    
    ncc_results = []
    
    for batch_idx in range(batch_size):
        # 获取单个batch的数据
        original_signal = original_signals[batch_idx, 0].flatten()
        watermarked_signal = watermarked_signals[batch_idx, 0].flatten()
        
        # 计算NCC
        ncc = calculate_ncc(original_signal, watermarked_signal)
        ncc_results.append(ncc)
    
    # 计算所有batch的平均NCC
    avg_ncc = np.mean(ncc_results)
    
    return avg_ncc


import numpy as np

def calculate_kcd(original_signal, watermarked_signal):
    """
    计算两个一维信号之间的KCD系数（假设为NCC）。
    """
    # 确保信号长度相同
    assert len(original_signal) == len(watermarked_signal), "Signals must have the same length."

    # 计算均值
    mean_original = np.mean(original_signal)
    mean_watermarked = np.mean(watermarked_signal)

    # 计算偏差
    deviation_original = original_signal - mean_original
    deviation_watermarked = watermarked_signal - mean_watermarked

    # 计算分子
    numerator = np.sum(deviation_original * deviation_watermarked)

    # 计算分母
    denominator = np.sqrt(np.sum(deviation_original ** 2)) * np.sqrt(np.sum(deviation_watermarked ** 2))

    # 避免除以零
    if denominator == 0:
        return 0

    # 返回KCD值
    return numerator / denominator

def calculate_batch_kcd(original_signals, watermarked_signals):
    """
    计算原始信号和嵌入水印后信号之间的KCD，并返回所有batch的平均值。
    """
    batch_size,  channels, samples = original_signals.shape
    
    kcd_results = []
    
    for batch_idx in range(batch_size):
        # 获取单个batch的数据
        original_signal = original_signals[batch_idx, 0].flatten()
        watermarked_signal = watermarked_signals[batch_idx, 0].flatten()
        
        # 计算KCD
        kcd = calculate_kcd(original_signal, watermarked_signal)
        kcd_results.append(kcd)
    
    # 计算所有batch的平均KCD
    avg_kcd = np.mean(kcd_results)
    
    return avg_kcd



from scipy.stats import entropy

# 定义Jensen-Shannon Divergence函数
def jensen_shannon_divergence(p, q):
    """
    计算两个概率分布p和q之间的JSD。
    """
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

# 定义将信号转换为概率分布的函数
def signal_to_distribution(signal):
    """
    将信号转换为概率分布。
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    bins = np.linspace(min_val, max_val, num=100)
    hist, _ = np.histogram(signal, bins=bins, density=True)
    return hist

# 定义计算JSD的函数，比较每个batch的数据
def calculate_batch_jsd(original_signals, watermarked_signals):
    """
    计算原始信号和嵌入水印后信号之间的JSD，并返回所有batch的平均值。
    """
    batch_size,  channels, samples = original_signals.shape
    
    jsd_results = []
    
    for batch_idx in range(batch_size):
        # 获取单个batch的数据
        original_signal = original_signals[batch_idx, 0]
        watermarked_signal = watermarked_signals[batch_idx, 0]
        
        # 转换为概率分布
        p = signal_to_distribution(original_signal)
        q = signal_to_distribution(watermarked_signal)
        
        # 计算JSD
        jsd = jensen_shannon_divergence(p, q)
        jsd_results.append(jsd)
    
    # 计算所有batch的平均JSD
    avg_jsd = np.mean(jsd_results)
    
    return avg_jsd



if __name__ == '__main__':
    # 创建模拟数据作为原始信号和水印嵌入后的信号
    batch_size = 5
    channels = 64
    samples = 1000

    original_signals = np.random.normal(loc=0, scale=1, size=(batch_size, 1, channels, samples))
    watermarked_signals = original_signals + np.random.normal(loc=0, scale=0.1, size=(batch_size, 1, channels, samples))

    # 计算JSD
    avg_jsd = calculate_batch_jsd(original_signals, watermarked_signals)

    print(f"Average Jensen-Shannon Divergence across all batches: {avg_jsd}")
    
    # 计算KCD
    avg_kcd = calculate_batch_kcd(original_signals, watermarked_signals)

    print(f"Average KCD across all batches: {avg_kcd}")
    
    # 计算NCC
    avg_ncc = calculate_batch_ncc(original_signals, watermarked_signals)
    print(f"Average NCC across all batches: {avg_ncc}")
    
    # 计算SSIM
    avg_ssim = calculate_batch_ssim(original_signals, watermarked_signals)

    print(f"Average SSIM across all batches: {avg_ssim}")
    
    
    # # 计算PSNR
    # avg_psnr = calculate_batch_psnr(original_signals, watermarked_signals)
    # print(f"Average PSNR across all batches: {avg_psnr:.2f} dB")