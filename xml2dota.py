import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

def convert_xml_to_txt(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有XML文件
    xml_files = [f for f in os.listdir(input_folder) if f.endswith(".xml")]

    # 使用tqdm遍历输入文件夹中的所有XML文件，并显示进度条
    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        xml_path = os.path.join(input_folder, xml_file)
        # 读取XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 构建输出txt文件路径
        txt_file = os.path.join(output_folder, os.path.splitext(xml_file)[0] + '.txt')
        # 打开txt文件进行写入
        with open(txt_file, 'w') as f:
            # 循环遍历每个object节点
            for obj in root.findall('.//object'):
                # 提取需要的信息
                x_left_top = obj.find('.//x_left_top').text
                y_left_top = obj.find('.//y_left_top').text
                x_right_top = obj.find('.//x_right_top').text
                y_right_top = obj.find('.//y_right_top').text
                x_right_bottom = obj.find('.//x_right_bottom').text
                y_right_bottom = obj.find('.//y_right_bottom').text
                x_left_bottom = obj.find('.//x_left_bottom').text
                y_left_bottom = obj.find('.//y_left_bottom').text
                name = obj.find('.//name').text
                angle = obj.find('.//angle').text
                # 写入到txt文件
                line = f"{x_left_top} {y_left_top} {x_right_top} {y_right_top} {x_right_bottom} {y_right_bottom} {x_left_bottom} {y_left_bottom} {name} {angle}\n"
                f.write(line)

    print("批量转换完成。")

if __name__ == "__main__":
    # 输入文件夹和输出文件夹路径
    input_folder = '/data/wangzerui/Code_yolo/ultralytics-8.2.52/datasets/DIOR-R/DIOPR-R-Completed/DIOR/Annotations/Oriented Bounding Boxes'
    output_folder = '/data/wangzerui/Code_yolo/ultralytics-8.2.52/datasets/DIOR-R/DIOPR-R-Completed/DIOR/Annotations/xml2dota'
    convert_xml_to_txt(input_folder, output_folder)
