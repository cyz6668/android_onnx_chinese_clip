# android_onnx_chinese_clip
## 1.模型介绍


Chinese-CLIP的核心在于其融合了自然语言处理和计算机视觉两个领域的最新进展。模型基于Transformer架构，通过大量的无标注图像-文本对进行预训练，学习图像特征和文本表示之间的对应关系。这种端到端的学习方式使得模型能够捕捉到丰富的视觉信息和深层次的语言含义。

## 2.项目介绍
![image](https://github.com/user-attachments/assets/f554b4f6-f7c2-454c-93dd-5b6c0fad59a7)

这里选用了以resnet50为backbone的cn-clip模型进行安卓不联网部署，实现以文搜图的简单应用demo。将chinese clip模型成功运行于手机。

## 3.主要工作
![image](https://github.com/user-attachments/assets/92f33500-a2d8-4062-b4d0-1dfed70d4e51)

1）pt模型转换为onnx；

2）onnxsim工具将模型简化；

3）运用jni技术处理预处理过程；这里预处理主要为上图中将中文文本经过tokenize处理，经过text encoder生成embedding向量，将图片经过transform之后经过image encoder生成embedding向量。

4）onnxruntime完成模型加载和推理。

5）完成输出后处理。将embedding向量做normalize处理。计算输入文字的embedding向量与全部图片的embedding向量的相似性。

6）比对多个图片不同的相似性，取最高者进行图片展示。
