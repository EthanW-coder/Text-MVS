# Text-MVS: Language Descriptions Text Dataset for Aviation Multi-view Stereo Reconstruction

#### Pre-prepared

To obtain the complete dataset, please first download the [LuoJia-MVS dataset](https://irsip.whu.edu.cn/resv2/resources_v2.php) and the [WHU dataset](https://gpcv.whu.edu.cn/data/WHU_MVS_Stereo_dataset.html).

#### Understand our dataset

The organizational structure of the dataset proposed in this paper is as follows:

```
├── MVS_TXT
│   ├── Template_A
│   │   ├── LuoJia_MVS_dataset
│   │   │   ├──test
│   │   │   ├──train
│   │   │   │   ├──txt
│   │   │   │   │   │   ├──4_52
│   │   │   │   │   │   │   ├──0
│   │   │   │   │   │   │   │   ├──000000.txt
│   │   └── WHU_MVS_dataset
│   │   │   ├──test
│   │   │   ├──train
│   │   │   │   ├──txt
│   │   │   │   │   │   ├──002_35
│   │   │   │   │   │   │   ├──0
│   │   │   │   │   │   │   │   ├──000000.txt
│   │   │   │   │   │   │   │   ├──000001.txt
│   │   │   │   │   │   │   │   ├──000002.txt
│   ├── Template_B
│   │   ├── LuoJia_MVS_dataset
│   │   └── WHU_MVS_dataset
│   ├── Template_C
│   │   ├── LuoJia_MVS_dataset
│   │   └── WHU_MVS_dataset
```

Our dataset shares the same organizational structure as the LuoJia-MVS and WHU datasets. To reduce computational load, LP-MVS only processes the text descriptions corresponding to the reference images.

#### Reproduce our language description data

Our text description inference model uses Qwen2.5-VL-7B. For details, please refer to [Qwen2.5-VL-7B-Instruct · Hugging Face](https://github.com/QwenLM/Qwen3-VL)

Execute the following command to load Qwen2.5-VL-7B into your project:

```
git clone https://github.com/QwenLM/Qwen2.5-VL.git
pip install -r requirements_web_demo.txt
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct
```

Our Inference Device: NVIDIA L20 (48GB VRAM) 

Our Deep Learning Framework: PyTorch 2.4.0

###### Note: While this specific hardware and software configuration was used for development, it is not an absolute requirement. The code is expected to be compatible with other modern GPUs and recent versions of PyTorch, though performance and memory usage may vary.

Run run_inference.py to infer the language description based on the prompt:

```python
python run_inference.py
```

Our prompt used for inference：

```
1. Describe this aerial image in one sentence, explicitly stating the core features in the image, their relative orientation and distance relationships, terrain undulation characteristics, and the overall geographic layout.

2. Please describe this aerial image in natural language as realistically as possible in one sentence, clearly identifying the main features in the image, their relative positions and distance relationships, as well as the characteristics of the terrain relief

3. Describe this aerial image in one sentence, analyze the size and arrangement of the features in the image, their relative positions, and interpret the depth of the scene through gradients in texture detail and sharpness.

4. Describe this aerial image in one sentence, outputting the scale and depth of each visible object, as well as the appearance details between them and their absolute positions in the overall image.
```

#### Citation

If you find this work useful in your research, please consider citing the following preprint:

```

```

#### Reference

This dataset is based on the implementations of LuoJia-MVS and WHU dataset. We thank them for providing the valuable source data in the field of Multi-view Stereo Reconstruction from Open Aerial imagery.

