import os
import argparse
import shutil

from matplotlib import pyplot as plt
import glob
import sys
import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image
import timm
import clip
import gradio as gr
from gradio.components import Image as GradioImage, Dropdown


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_feature_by_model(model, preprocess, file, model_name, input_size=128):
    """
    提取给定文件的特征向量。

    参数:
    model: 用于提取特征的模型。
    preprocess: 预处理函数，适用于特定的模型。
    file: 要处理的图像文件路径。
    model_name: 使用的模型名称，用于决定特征提取的方法。
    input_size: 输入图像的大小，仅在非 CLIP/LLaVA 模型中使用。

    返回:
    提取的特征向量。
    """
    img_rgb = Image.open(file).convert('RGB')

    if model_name == "clip":
        image = preprocess(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(image)
    elif model_name == "LLaVA":
        # 如果有 LLaVA 的特定处理方式，请在这里添加
        pass
    else:
        # 对于其他 TIMM 模型的处理
        image = img_rgb.resize((input_size, input_size), Image.LANCZOS)
        image = torchvision.transforms.ToTensor()(image)
        trainset_mean = [0.47083899, 0.43284143, 0.3242959]
        trainset_std = [0.37737389, 0.36130483, 0.34895992]
        image = torchvision.transforms.Normalize(mean=trainset_mean, std=trainset_std)(image).unsqueeze(0)
        with torch.no_grad():
            features = model.forward_features(image)
            vec = model.global_pool(features)

    vec = vec.squeeze().cpu().numpy()
    img_rgb.close()
    return vec



def extract_features(args, model, image_path='', preprocess=None):
    """
    从指定路径提取所有图像的特征向量。

    参数:
    args: 包含配置参数的对象。
    model: 用于提取特征的模型。
    image_path: 包含图像的目录路径。
    preprocess: 预处理函数，适用于特定的模型。

    返回:
    包含所有图像及其特征向量的字典。
    """
    allVectors = {}  # 对数据库中所有照片的表征存储

    for image_file in tqdm.tqdm(glob.glob(os.path.join(image_path, '*', '*.jpg'))):
        # 使用新的封装函数
        allVectors[image_file] = extract_feature_by_model(model, preprocess, image_file, args.model_name)

    os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok=True)

    np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", allVectors)

    return allVectors


def getSimilarityMatrix(vectors_dict):
    """
    计算特征向量之间的相似度矩阵。

    参数:
    vectors_dict: 包含特征向量的字典。

    返回:
    相似度矩阵及对应的键列表。
    """
    v = np.array(list(vectors_dict.values()))  # [NUM, H],把所有表征向量转化成数组

    numerator = np.matmul(v, v.T)  # [NUM, NUM]
    # 对向量的第一维度求一个范数，同时维度保持不变
    denominator = np.matmul(np.linalg.norm(v, axis=1, keepdims=True),
                            np.linalg.norm(v, axis=1, keepdims=True).T)  # [NUM,NUM]

    sim = numerator / denominator
    keys = list(vectors_dict.keys())
    return sim, keys


def setAxes(ax, image, query=False, **kwargs):
    """
    设置绘图轴的属性。

    参数:
    ax: matplotlib 轴对象。
    image: 图像文件名。
    query: 是否为查询图像。
    kwargs: 其他可选参数，例如分数值。
    """
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel("Query Image\n{0}".format(image), fontsize=12)
        ax.xaxis.label.set_color('red')
    else:
        ax.set_xlabel("score = {1:1.3f}\n{0}".format(image, value), fontsize=12)
        ax.xaxis.label.set_color('blue')

    ax.set_xticks([])
    ax.set_yticks([])


def plotSimilarImages(args, image, simImages, simValues, numRow=1, numCol=4):
    """
    绘制相似图像及其得分。

    参数:
    args: 包含配置参数的对象。
    image: 查询图像文件路径。
    simImages: 相似图像的列表。
    simValues: 相似图像的得分列表。
    numRow: 图像网格的行数。
    numCol: 图像网格的列数。
    """
    fig = plt.figure()

    # set width and height in inches
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(f"use engine model: {args.model_name}", fontsize=35)

    for j in range(0, numCol * numRow):
        ax = []
        if j == 0:  # query照片Axes设置不太一样
            img = Image.open(image)
            ax = fig.add_subplot(numRow, numCol, 1)
            setAxes(ax, image.split(os.sep)[-1], query=True)
        else:
            img = Image.open(simImages[j - 1])
            ax.append(fig.add_subplot(numRow, numCol, j + 1))
            setAxes(ax[-1], simImages[j - 1].split(os.sep)[-1], value=simValues[j - 1])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()

    fig.savefig(
        f"{args.save_dir}/{args.model_name}_search_top{args.topk}_{image.split(os.sep)[-1].split('.')[0]}.png")  # 照片存入磁盘
    plt.show()

def save_uploaded_image(uploaded_filepath, save_dir="/Users/twosugar/Desktop/dataset_fruit_veg/upload"):
    """
    将上传的图像文件保存到指定目录。

    参数:
    uploaded_filepath: 上传的图像文件路径。
    save_dir: 保存图像的目录路径。

    返回:
    保存的图像文件路径。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 计算目标路径
    target_path = os.path.join(save_dir, os.path.basename(uploaded_filepath))

    # 复制文件到目标路径
    shutil.copy(uploaded_filepath, target_path)

    return target_path

def search_similar_images(uploaded_image, selected_model):
    """
    搜索上传图像的相似图像。

    参数:
    uploaded_image: 上传的图像文件路径。
    selected_model: 选择的模型名称。

    返回:
    相似图像的列表。
    """
    # 确保 selected_model 是字符串
    if isinstance(selected_model, list):
        selected_model = selected_model[0]
    # 定义必要的参数
    input_size = 128  # 或者根据需要设置
    topk = 7  # 或者根据需要设置
    save_dir = "./output_dir"  # 或者根据需要设置
    feature_dict_file = "corpus_feature_dict.npy"  # 或者根据需要设置

    # 根据 selected_model 加载模型
    if selected_model == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device)
    else:
        model = timm.create_model(selected_model, pretrained=True)
        model.eval()
        preprocess = None  # 如果需要，为 TIMM 模型设置适当的预处理

    # 保存上传的图像并提取其特征
    image_path = save_uploaded_image(uploaded_image)
    query_features = extract_feature_by_model(model, preprocess, image_path, selected_model)

    # 加载特征字典并计算相似度
    allVectors = np.load(f"{save_dir}/{selected_model}/{feature_dict_file}", allow_pickle=True).item()
    allVectors[image_path] = query_features

    sim, keys = getSimilarityMatrix(allVectors)
    index = keys.index(image_path)
    sim_vec = sim[index]

    # 获取最相似的图像
    indexs = np.argsort(sim_vec)[::-1][1:topk + 1]
    simImages = [keys[ind] for ind in indexs]

    return simImages


model_names = timm.list_models(pretrained=True) + ["clip"]  # 假设你想包括所有 TIMM 模型和 CLIP 模型

# 确保 Gradio 界面中使用的是 GradioImage
iface = gr.Interface(
    fn=search_similar_images,
    inputs=[
        GradioImage(type="filepath", label="Upload Image"),
        Dropdown(choices=model_names, label="Select Model")
    ],
    outputs="gallery",
    title="Image Search Engine",
    description="Upload an image and select a model to search for similar images."
)

if __name__ == '__main__':
    iface.launch()