import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

def bgr_to_hsv(bgr_2dim):
    normalized_bgr_2dim = np.array(bgr_2dim) / 255.0
    b, g, r = normalized_bgr_2dim[:,0], normalized_bgr_2dim[:,1], normalized_bgr_2dim[:,2]
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val

    # Initialize h, s, and v with zeros
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    v = max_val * 100  # Value

    # Compute Hue
    cond_r = max_val == r
    cond_g = max_val == g
    cond_b = max_val == b
    cond_mx_mn = max_val == min_val

    h = np.where(cond_r, 60 * ((g - b) / diff % 6), h)
    h = np.where(cond_g, 60 * ((b - r) / diff + 2), h)
    h = np.where(cond_b, 60 * ((r - g) / diff + 4), h)
    h = np.where(cond_mx_mn, 0, h)

    # Compute Saturation
    s = np.where(max_val != 0, (diff / max_val) * 100, 0)

    # Stack them into a 2D array
    hsv_2dim = np.stack([h, s, v], axis=1)
    return hsv_2dim

def hsv_to_rgb(hsv_2dim, norm_type="cv2"):
    hsv_2dim = np.array(hsv_2dim).astype(np.float32)

    if norm_type == "cv2":
        scale = np.array([179.0,255.0,255.0])
    else:
        scale = np.array([360.0,100.0,100.0])

    normalized_hsv_2dim = hsv_2dim / scale
    return mcolors.hsv_to_rgb(normalized_hsv_2dim) * 255.0
    
def hsv_to_bgr(hsv_2dim):
    rgb_2dim = hsv_to_rgb(hsv_2dim)
    bgr_2dim = np.array([list(reversed(rgb_1dim)) for rgb_1dim in rgb_2dim])
    return bgr_2dim


def visualize_rgb_colors(rgb_colors):
    rgb_colors = np.array(rgb_colors)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Loop through the list of BGR colors and plot each one
    for i, color in enumerate(rgb_colors):
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_color = [x / 255.0 for x in color]

        # Create a rectangle filled with the normalized RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(rgb_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()
    
def make_pil_rgb_colors(rgb_colors):
    rgb_colors = np.array(rgb_colors)

    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Loop through the list of BGR colors and plot each one
    for i, color in enumerate(rgb_colors):
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_color = [x / 255.0 for x in color]

        # Create a rectangle filled with the normalized RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(rgb_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Get the RGB image as a NumPy array
    fig.canvas.draw()
    rgb_image_np = np.array(fig.canvas.renderer.buffer_rgba())

    # Close the plot to free up resources
    plt.close()

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(rgb_image_np)

    # Return the PIL Image
    return pil_image

def visualize_bgr_colors(bgr_colors):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Loop through the list of BGR colors and plot each one
    for i, color in enumerate(bgr_colors):
        # Convert BGR to RGB and normalize to [0, 1]
        rgb_color = [x / 255.0 for x in reversed(color)]

        # Create a rectangle filled with the normalized RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color)
        
        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(bgr_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()


def visualize_hsv_colors(hsv_colors, norm_type=None):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    rgb_colors = hsv_to_rgb(hsv_colors, norm_type=norm_type)
    # Loop through the list of HSV colors and plot each one
    for i, rgb_color in enumerate(rgb_colors):
        # Create a rectangle filled with the RGB color
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=rgb_color/255)

        # Add the rectangle to the plot
        ax.add_patch(rect)

    # Set axis limits and aspect ratio
    ax.set_xlim(0, len(hsv_colors))
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Remove axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the plot
    plt.show()

# clustering kmeans
def img_none_flatten(img, mask):
    if mask is not None:
        img = np.where(mask > 127, img, np.nan)
    return img[~np.isnan(img[:, :, :3]).any(axis=2)]

class Point:
    def __init__(self, data, K=4):
        self.data = data
        self.k = np.random.randint(0, K)

    def __repr__(self):
        return str({"data": self.data, "k": self.k})

def make_k_mapping(points):
    point_dict = defaultdict(list)
    for p in points:
        point_dict[p.k] = point_dict[p.k] + [p.data]
    return point_dict


def calc_k_means(point_dict, K=4):
    means = [np.mean(point_dict[k], axis=0) for k in range(K)]
    return means


def update_k(points, means, K=4):
    for p in points:
        dists = [np.linalg.norm(means[k] - p.data) for k in range(K)]
        p.k = np.argmin(dists)


def find_color_type_fixed(img, mask=None, n_clusters=4, epochs=1):
    img_flatten = img_none_flatten(img.copy(), mask.copy())
    points = [Point(d, K=n_clusters) for d in img_flatten]
    point_dict = make_k_mapping(points)
    colors = calc_k_means(point_dict, K=n_clusters)
    update_k(points, colors, K=n_clusters)
    for e in range(epochs):
        point_dict = make_k_mapping(points)
        colors = calc_k_means(point_dict, K=n_clusters)
        update_k(points, colors, K=n_clusters)

    try:
        colors = [[int(c) for c in color] if isinstance(color, np.ndarray) or isinstance(color, list) else [0, 0, 0] for color in colors]
    except TypeError:
        print(colors)
        raise TypeError

    percentage = [0 for _ in range(n_clusters)]
    for p in points:
        percentage[p.k] += 1
    percentage = [p / len(img_flatten) for p in percentage]

    return colors, percentage


def preprocess_input_image(input_image):
    return input_image

def sort_colors(colors):
    colors.sort(key=lambda c: c[2])
    colors.sort(key=lambda c: c[1])
    colors.sort(key=lambda c: c[0])
    return colors

def sort_colors_hsv(colors):
    if not isinstance(colors, list):
        colors = colors.tolist()
        
    colors.sort(key=lambda c: c[2]**2 + c[1]**2, reverse=True)
    return colors

def sort_color_feature_mean_dist(colors):
    colors = np.array(colors)
    mean_color = np.mean(colors, axis=0)
    dist = np.sum((colors - np.array([mean_color for _ in range(len(colors))])) ** 2, axis=1)/len(colors)
    
    return colors[np.argsort(dist)[::-1]].tolist()


# Kmeans re implementation

class Centroid:
    def __init__(self, color):
        self.data = color
        self.neighbors = np.array([[127,127,127]])

    def update_color(self):
        if len(self.neighbors) == 0:
            self.data = np.array([-255,-255,-255])
        self.data = np.mean(self.neighbors, axis=0)

    def get_dist(self, t_color):
        dist = np.linalg.norm(self.data - t_color)
        return dist
    
    def add_neighbor(self, color):
        self.neighbors = np.append(self.neighbors, color.reshape([1,-1]), axis=0)
    
    def clean_neighbor(self):
        self.neighbors = self.data[None,:]
        
    def get_data(self):
        return self.data.astype(np.int16).tolist(), len(self.neighbors)
    
def random_selected_pixel_with_mask(img, mask=None, select_n=4):
    if mask.all() == None:
        mask = np.ones_like(img[:,:,0])
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    if len(np.unique(mask)) != 2:
        mask = np.where(mask<127, 0, 1)

    img_flat = img.reshape([-1,3])
    mask_flat = mask.flatten()

    selected_colors = np.array([])

    count = 0
    while(len(np.unique(selected_colors, axis=0)) != select_n):
        selected_pixels = np.random.choice(np.where(mask_flat == 1)[0], select_n, replace=False)    
        selected_colors = img_flat[selected_pixels]
        
        count += 1

        if count >= 30:
            return np.array([[0,0,0],[255,0,0],[0,255,0],[0,0,255]])

    return selected_colors

def color_filter_with_mask(img, mask, pixel_skip):
    if mask.all() == None:
        mask = np.ones_like(img[:,:,0])
    if len(mask.shape) == 3:
        mask = mask[:,:,0]
    if len(np.unique(mask)) != 2:
        mask = np.where(mask<127, 0, 1)

    img_flat = img.reshape([-1,3])
    mask_flat = mask.flatten()
    img_flat = img_flat[np.where(mask_flat == 1)[0]]
    random_row = np.random.choice(len(img_flat), size=len(img_flat)//pixel_skip, replace=True)

    return img_flat[random_row,:]


def color_extraction(img, mask=None, n_cluster=4, epochs = 3, pixel_skip=10, per_round=None):
    if mask is None:
        mask = np.ones_like(img) * 255

    img = color_filter_with_mask(img, mask, pixel_skip)

    # selected_colors = random_selected_pixel_with_mask(img, mask, n_cluster)
    if len(img) == 0:
        print(np.unique(mask))
    selected_colors = img[np.random.choice(range(len(img)), n_cluster)]
    k_map = {idx:Centroid(color) for idx, color in enumerate(selected_colors)}

    result = []
    for epoch in range(epochs):
        for color in img:
            closest_k = np.argmin([k_map[k].get_dist(color) for k in range(n_cluster)])
            k_map[closest_k].add_neighbor(color)

        for k in range(n_cluster):
            k_map[k].update_color()

            if epoch == epochs-1:
                color, num_pix = k_map[k].get_data()
                result.append((color, num_pix/len(img)))
            else:
                k_map[k].clean_neighbor()

    result = sorted(result, key=lambda x:x[1], reverse=True)

    color_result = []
    percentage = []
    for cur in result:
        color_result.append(cur[0])
        if per_round is not None:
            per = round(cur[1], per_round)
        else:
            per = cur[1]
        percentage.append(per)

    return [color_result, percentage]



def color_normalization(color_arr, scaling = True, type="rgb", only_scale=False):

    color_arr = np.array(color_arr).astype(np.float32)

    if type == "rgb":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        scale = np.array([255.0, 255.0, 255.0])
    elif type == "hsv":
        mean = np.array([0.314 , 0.3064, 0.553])
        std = np.array([0.2173, 0.2056, 0.2211])
        scale = np.array([179.0, 255.0, 255.0])
        
    arr_shape_length = len(color_arr.shape)    
    new_shape = [1]*arr_shape_length
    new_shape[arr_shape_length-1] = -1

    mean = mean.reshape(new_shape)
    std = std.reshape(new_shape)
    scale = scale.reshape(new_shape)

    if scaling == True:
        color_arr /= scale

    if only_scale == True:
        return color_arr
    
    return (color_arr - mean)/std


def color_normalization_restore(color_arr, scaling = True, type="rgb"):
    color_arr = np.array(color_arr).astype(np.float32)

    if type == "rgb":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        scale = np.array([255.0, 255.0, 255.0])
    elif type == "hsv":
        mean = np.array([0.314 , 0.3064, 0.553])
        std = np.array([0.2173, 0.2056, 0.2211])
        scale = np.array([179.0, 255.0, 255.0])
        
    arr_shape_length = len(color_arr.shape)
    new_shape = [1]*arr_shape_length
    new_shape[arr_shape_length-1] = -1
 
    mean = mean.reshape(new_shape)
    std = std.reshape(new_shape)
    scale = scale.reshape(new_shape)

    restored_arr = color_arr * std + mean

    if scaling == True:
        restored_arr *= scale

    return restored_arr

import cv2
def rgb2hsv_cv2(rgb_colors):
    rgb_colors = np.array(rgb_colors, dtype=np.uint8)

    input_shape_len = len(rgb_colors.shape)
    if input_shape_len < 3:
        rgb_colors = rgb_colors.reshape(1,-1,3)
    
    hsv_colors = cv2.cvtColor(rgb_colors, cv2.COLOR_RGB2HSV).astype(np.uint8)

    return hsv_colors.squeeze()

def hsv2rgb_cv2(hsv_colors):
    hsv_colors = np.array(hsv_colors, dtype=np.uint8)

    input_shape_len = len(hsv_colors.shape)
    if input_shape_len < 3:
        hsv_colors = hsv_colors.reshape(1,-1,3)
    
    rgb_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2RGB).astype(np.uint8)

    return rgb_colors.squeeze()


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

def colors_to_hex(colors):
    colors = list(colors)
    return [rgb_to_hex(color) for color in colors]


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / max(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


from PIL import Image

import json
import torch
import numpy as np
from PIL import Image
from typing import Union

def prepare_latent_width_height(pil_image_list=None, explicitly_define_size:Union[list, None]=None, vae_scale=8):
    if explicitly_define_size:
        WIDTH, HEIGHT = explicitly_define_size
        LATENTS_WIDTH = (WIDTH // vae_scale) + (-(WIDTH // vae_scale) % vae_scale if (WIDTH // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (WIDTH // vae_scale) % vae_scale))
        LATENTS_HEIGHT = (HEIGHT // vae_scale) + (-(HEIGHT // vae_scale) % vae_scale if (HEIGHT // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (HEIGHT // vae_scale) % vae_scale))
        return WIDTH, HEIGHT, LATENTS_WIDTH*vae_scale, LATENTS_HEIGHT*vae_scale, LATENTS_WIDTH, LATENTS_HEIGHT

    image_sizes = []
    for pil_image in pil_image_list:
        if isinstance(pil_image, list):
            for cur in pil_image:
                if pil_image is not None:
                    image_sizes.append(cur.size)
        else:
            if pil_image is not None:
                image_sizes.append(pil_image.size)
    
    if len(image_sizes) == 0:
        WIDTH, HEIGHT = (512, 512)
        LATENTS_WIDTH = (WIDTH // vae_scale) + (-(WIDTH // vae_scale) % vae_scale if (WIDTH // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (WIDTH // vae_scale) % vae_scale))
        LATENTS_HEIGHT = (HEIGHT // vae_scale) + (-(HEIGHT // vae_scale) % vae_scale if (HEIGHT // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (HEIGHT // vae_scale) % vae_scale))
        return WIDTH, HEIGHT, LATENTS_WIDTH*vae_scale, LATENTS_HEIGHT*vae_scale, LATENTS_WIDTH, LATENTS_HEIGHT

    if not all(size == image_sizes[0] for size in image_sizes):
        raise ValueError("All image must have same size")
            
    WIDTH, HEIGHT = image_sizes[0]
    LATENTS_WIDTH = (WIDTH // vae_scale) + (-(WIDTH // vae_scale) % vae_scale if (WIDTH // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (WIDTH // vae_scale) % vae_scale))
    LATENTS_HEIGHT = (HEIGHT // vae_scale) + (-(HEIGHT // vae_scale) % vae_scale if (HEIGHT // vae_scale) % vae_scale < vae_scale/2 else (vae_scale - (HEIGHT // vae_scale) % vae_scale))

    return WIDTH, HEIGHT, LATENTS_WIDTH*vae_scale, LATENTS_HEIGHT*vae_scale, LATENTS_WIDTH, LATENTS_HEIGHT

def check_prompt_text_length(prompt_list, max_length=77):
    truncated_prompt_list = []
    for prompt in prompt_list:
        if isinstance(prompt, str):
            if len(prompt) > max_length:
                print(f"prompts is too long it will be truncated to {max_length} len")
            truncated_prompt_list.append(prompt[:max_length])
        else:
            truncated_prompt_list.append(prompt)
    return truncated_prompt_list

def rescale(x, old_range, new_range, clamp=False, out_type="pt"):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    
    x = x.to(dtype=torch.float16)

    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)

    if out_type == "np":
        x = np.array(x)
    return x

def get_time_embedding(timestep, dtype=torch.float16):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160) / 160)
    # Shape: (1, 160)
    if len(timestep.shape) == 0:
        timestep = torch.tensor([timestep])
    x = timestep.clone().detach()[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1).to(dtype) 

def get_model_weights_dtypes(models_wrapped_dict, verbose=False):
    
    model_dtype_map = {}
    for name, model in models_wrapped_dict.items():
        if isinstance(model, list):
            model = model[0]
        first_param = next(model.parameters())
        model_dtype_map[name] = first_param.dtype
    
    dtype_list = list(model_dtype_map.values())
    
    if verbose == True:
        if all(x == dtype_list[0] for x in dtype_list):
            print(f"Models have same precision")
        else:
            print(f"Models are mixed precision")
            print(model_dtype_map)

    return model_dtype_map

def extract_euclidien_similarity(data_arr):
    data_arr = np.array(data_arr)
    norm_data = np.sum(data_arr ** 2, axis=1).reshape(-1, 1)
    squared_distances = norm_data + norm_data.T - 2 * np.dot(data_arr, data_arr.T)
    squared_distances = np.maximum(squared_distances, 0)
    distances = np.sqrt(squared_distances)
    similarities = 1 / (1 + distances)
    np.fill_diagonal(similarities, 1)
    
    return similarities

def get_colors_and_ids(palette, color_list):
    palette = np.array(palette)
    color_array = np.array(color_list)+1

    rgb_results = []
    id_results = []
    for target in palette:
        similarities = extract_euclidien_similarity(np.concatenate([target[None,:], color_array]))[0][1:]

        id_results.append(np.argmax(similarities))
        rgb_results.append((color_array[np.argmax(similarities)]-1).tolist())
        
    return rgb_results, id_results

def load_colot_list_data(path="./data/list_of_colors.jsonl"):
    color_list = {}
    with open(path, mode="r") as file:
        for line in file:
            line = json.loads(line)
            color_list[line["color_number"]] = line["color_rgb"]
    color_list = [cur[1] for cur in sorted(color_list.items(), key=lambda x:x[0])]
    return color_list

def composing_image(img1, img2, mask, out_type="pil"):
    img1 = np.array(img1)
    mask = np.array(mask)
    img2 = np.array(img2)
    
    composed_output = img1 * (1-rescale(mask, (np.min(mask), np.max(mask)), (0,1), out_type="np")) + \
        img2 * rescale(mask, (np.min(mask), np.max(mask)), (0,1), out_type="np")
    
    if out_type == "pil":
        composed_output = Image.fromarray(composed_output.astype(np.uint8))
    
    return composed_output

import base64, json
from io import BytesIO
from PIL import Image

import cv2
import numpy as np

class TooMuchRequestQueueError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return f"TooMuchRequestQueueError: {self.message}"

class DupledRequestKeyError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return f"DupledRequestKeyError: {self.message}"

class RequestKeyDoesNotExistError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return f"RequestKeyDoesNotExistError: {self.message}"
    
def respond(err, res):
    respond_msg = {'statusCode': 502 if err is not None else 200, 'body': json.dumps(res)}
    return respond_msg

def bs64_to_pil(img_bs64):
    if not img_bs64.startswith("/"):
        img_bs64 = img_bs64.split(",", 1)[1]
    img_data = base64.b64decode(img_bs64)
    img_pil = Image.open(BytesIO(img_data))
    return img_pil

def pil_to_bs64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="jpeg")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def img_box_crop(img_pil, box):
    if isinstance(box, list):
        x1, y1, x2, y2 = box
    else:
        x1 = box["x1"]; y1 = box["y1"]
        x2 = box["x2"]; y2 = box["y2"]
    
    return img_pil.crop((x1, y1, x2, y2))

def padding_mask_img(img_pil, mask_pil, box):
    if box == None:
        return mask_pil
    
    if isinstance(box, list):
        x1, y1, x2, y2 = box
    else:
        x1 = box["x1"]; y1 = box["y1"]
        x2 = box["x2"]; y2 = box["y2"]
    
    black_img = Image.new("RGB", img_pil.size)
    black_img.paste(mask_pil, (x1, y1, x2, y2))
    return black_img

def load_instance_from_json(json_like):
    args = json_like["body"]
    if isinstance(args, str):
        args = json_like.loads(args)
    return args

def center_crop_and_resize(input_image, target_size=(512, 512)):
    image = input_image
    width, height = image.size
    left = (width - min(width, height)) // 2
    top = (height - min(width, height)) // 2
    right = (width + min(width, height)) // 2
    bottom = (height + min(width, height)) // 2
    image = image.crop((left, top, right, bottom))

    image = image.resize(target_size)

    return image

def composing_output(img1, img2, mask):
    img1 = np.array(img1)
    mask = np.array(mask)
    img2 = np.array(img2)
    
    composed_output = np.array(img1) * (1-mask/255) + np.array(img2) * (mask/255)
    return Image.fromarray(composed_output.astype(np.uint8))

def make_canny_condition(image, min=100, max=200):
    image = np.array(image)
    image = cv2.Canny(image, min, max)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

def make_outpaint_condition(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    black_image = np.zeros_like(image)

    composed_output = np.array(black_image) * (1-mask/255) + np.array(image) * (mask/255)
    return Image.fromarray(composed_output.astype(np.uint8))

def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise

def make_shuffle_condition(image, h=None, w=None, f=None):
    img = np.array(image)
    H, W, C = img.shape
    if h is None:
        h = H
    if w is None:
        w = W
    if f is None:
        f = 256
    x = make_noise_disk(h, w, 1, f) * float(W - 1)
    y = make_noise_disk(h, w, 1, f) * float(H - 1)
    flow = np.concatenate([x, y], axis=2).astype(np.float32)
    return Image.fromarray(cv2.remap(img, flow, None, cv2.INTER_LINEAR))
    
def resize_store_ratio(image, min_side=512):

    width, height = image.size

    if width < height:
        new_width = min_side
        new_height = int((height / width) * min_side)
    else:
        new_width = int((width / height) * min_side)
        new_height = min_side

    resized_image = image.resize((new_width, new_height))

    return resized_image

def make_background(image_size, color_rgb=[255,255,255], noise=0.1):
    w, h = image_size
    image_array = np.zeros((h, w, 3), dtype=np.uint8)
    image_array[:, :] = color_rgb

    # Add noise to the image
    noise_pixels = np.random.randint(0, 256, (h, w, 3))
    image_array = (1 - noise) * image_array + noise * noise_pixels
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    image = Image.fromarray(image_array, 'RGB')
    return image
