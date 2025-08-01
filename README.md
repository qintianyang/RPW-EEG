# 触发集水印制作

水印的满足信息

编码器采用dvmark 解码器是hidden

触发集的满足条件是，身份信息不变、但是任务分类被破坏

loss 包含了 对抗损失、水印损失、脑电损失

# 生成的触发集返回四个数值

生成的虚假脑电

''''
wrong_predictions.append(
        imgs[i].cpu(),  # 输入数据
        task_labels[i].item(),  # 真实标签
        preds[i].item(),  # 预测标签
        identify_id[i].item() #身份的标签
    )
''''