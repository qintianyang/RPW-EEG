# RPW-EEG: An Unified Framework for Robust and Practical Watermark of EEG  
A robust and practical EEG watermarking framework for copyright protection and privacy preservation.  


## 🌟 Project Overview  
With the rapid expansion of brain-computer interfaces (BCIs) into consumer industries driven by metaverse and extended reality (XR) technologies, the privacy and security of EEG data have become critical concerns. EEG data, as sensitive biological signals carrying personal mental, cognitive, and health information, requires reliable copyright traceability and usability preservation .  

**RPW-EEG** is an end-to-end framework designed to address these challenges by embedding copyright information into EEG data as perturbations. It balances watermark robustness, data usability, and copyright traceability through a dual-stage training strategy, outperforming existing baselines in key metrics .  


## 🎯 Key Features  
- **Robust Watermarking**: Incorporates a noise layer to simulate attacks (time/frequency/spatial domain transformations) during training, enhancing anti-attack capabilities .  
- **Practical Usability**: Uses a plug-and-play adversarial fine-tuning module to restore task-related features (e.g., motor imagery), ensuring EEG data remains effective for downstream tasks .  
- **End-to-End Design**: Combines encoder-decoder architecture with dual-stage training (encoder-decoder training + adversarial fine-tuning) for seamless copyright embedding and extraction .  


## 🛠️ Framework Architecture  
RPW-EEG’s implementation consists of two core stages, as illustrated in Figure 1 of the paper:  



### 1. Encoder-Decoder Training  
- **Encoder**: Designed to embed copyright information into EEG data using transition layers (convolution, batch normalization, ReLU) and multi-scale watermark embedding layers. It preserves original EEG features via MSE loss .  
- **Noise Layer**: Simulates real-world attacks (e.g., Gaussian noise, time reversal, frequency shifting) to enhance watermark robustness .  
- **Decoder**: Extracts copyright information from watermarked EEG data using convolutional layers and global average pooling, optimized via L2 norm loss for accurate recovery .  

### 2. Adversarial Task Fine-Tuning  
A plug-and-play module that refines the watermark into an "adversarial watermark" by restoring task-specific features (e.g., motor imagery patterns). It freezes pre-trained task classifiers and updates the encoder to preserve task relevance, ensuring usability .  



## 🚀 Getting Started  
### Requirements  
- Python 3.x  
- PyTorch  
- NVIDIA GPU (e.g., RTX 3090) for training .  

### Installation  
The complete code is open-sourced and available at:  
[GitHub Repository](https://github.com/qintianyang/RPW-EEG) .  

```bash  
git clone https://github.com/qintianyang/RPW-EEG.git  
cd RPW-EEG  
pip install -r requirements.txt  
```  

### Usage  
1. **Data Preparation**: Download the datasets (Dataset I and Dataset II) as specified in the paper.  
2. **Training**: Run the dual-stage training pipeline (encoder-decoder training + fine-tuning) with default parameters (binary code length = 30 bits, learning rate = 1e-3) .  
3. **Evaluation**: Use provided scripts to test watermark quality (PSNR/SSIM), usability (task accuracy), and robustness (attack simulations).  




## 🙏 Acknowledgements  
Supported by:  
- National Natural Science Foundation of China (62471169)  
- Key Research and Development Project of Zhejiang Province (2023C03026, 2021C03001, 2021C03003)  
- Key Laboratory of Brain Machine Collaborative Intelligence of Zhejiang Province (2020E10010) .  


## 📄 Citation  
If you use this framework, please cite the original paper:  

Qin, T., Yi, H., Qian, J., et al. (2025). RPW-EEG: An Unified Framework for Robust and Practical Watermark of EEG. *Proceedings of the Annual Meeting of the Cognitive Science Society, 47(0)*. [https://escholarship.org/uc/item/96h9c291](https://escholarship.org/uc/item/96h9c291) .