import os
import shutil
import PIL.Image
from matplotlib import pyplot as plt
import glob
import tqdm
import numpy as np
import torch
import torchvision
from PIL import Image
import timm
import clip
import gradio as gr
import requests
from gradio.components import Image as GrImage, Dropdown, Textbox
import google.generativeai as genai


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GOOGLE_API_KEY = "AIzaSyBd3WntReNYp0QgUNYs8ntVOkZHznk32zs"
genai.configure(api_key=GOOGLE_API_KEY)

def call_geminipro_api(image_path, api_key):
    # 根据 GeminiPro API 的要求设置 URL 和请求参数
    url = "https://makersuite.google.com/app/prompts/new_freeform"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    files = {'image': open(image_path, 'rb')}

    response = requests.post(url, headers=headers, files=files)
    try:
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()  # 将触发异常，如果状态码不是 200-399
        return response.json()
    except requests.exceptions.HTTPError as errh:
        print("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print("Oops: Something Else",err)

def extract_feature_by_model(model, preprocess, file, model_name, input_size=128):
    img_rgb = Image.open(file).convert('RGB')

    if model_name == "clip":
        image = preprocess(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(image)
    elif model_name == "LLaVA":
        image = preprocess(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(image)
    else:
        # 对于其他模型，假设使用了单一的特征提取方法
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

def get_similarity_matrix(vectors_dict):
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

    sim, keys = get_similarity_matrix(allVectors)
    index = keys.index(image_path)
    sim_vec = sim[index]

    # 获取最相似的图像
    indexs = np.argsort(sim_vec)[::-1][1:topk + 1]
    simImages = [keys[ind] for ind in indexs]

    return simImages


def describe_image(uploaded_image, selected_model):
    # 调用 GeminiPro API
    selected_model = genai.GenerativeModel('gemini-pro-vision')
    # 获取图像路径
    image_path = save_uploaded_image(uploaded_image)
    # 打开图像文件
    img = PIL.Image.open(image_path)
    # 使用 Gemini Pro 对图像进行描述
    try:
        response = selected_model.generate_content(["你是一个图像分析专家,你需要将提供的图像使用中文描述出来", img], stream=True)
        response.resolve()
        description = response.text
    except Exception as e:
        description = f"Error in describing image with Gemini Pro: {e}"
    return description


model_names_1 = ["clip", "resnet50", "resnet152"]
model_names_2 = ["Gemini Pro"]


# 示例图像的路径和模型名称
examples = [
    ["/Users/twosugar/Desktop/Coding/ImageSearch/test/examples/example_1.png", "clip"],
    ["/Users/twosugar/Desktop/Coding/ImageSearch/test/examples/example_2.png", "resnet50"],
    ["/Users/twosugar/Desktop/Coding/ImageSearch/test/examples/example_3.png", "clip"],
]


# 为每个函数创建一个独立的接口
describe_interface = gr.Interface(
    fn=describe_image,
    inputs=[GrImage(type="filepath", label="Upload Image for Description"),
            Dropdown(choices=model_names_2, label="Select Model for Description")],
    outputs=Textbox(label="Image Description")
)

search_interface = gr.Interface(
    fn=search_similar_images,
    inputs=[GrImage(type="filepath", label="Upload Image for Similar Images"),
            Dropdown(choices=model_names_1, label="Select Model for Similar Images")],
    outputs="gallery"
)

demo = gr.TabbedInterface([describe_interface, search_interface], ["Describe", "Search"])

if __name__ == '__main__':
    demo.launch()