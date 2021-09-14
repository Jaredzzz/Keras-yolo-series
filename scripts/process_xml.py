# endcoding: utf-8
import numpy as np
import os
import xml.etree.ElementTree as ET
from skimage import io


def insert_node(annotation_path, save_path):
    for xml_name in os.listdir(annotation_path):
        if xml_name.endswith('xml'):
            print(xml_name)
            xml_file = os.path.join(annotation_path, xml_name)
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = xml_name.replace("xml", "jpg")
            # root.find('annotation').tail = root.find('annotation').tail + "\t"
            name = ET.SubElement(root, "filename")
            name.text = filename
            name.tail = "\n\t"
            tree = ET.ElementTree(root)
            tree.write(os.path.join(save_path, xml_name[:-4]) + ".xml", encoding="utf-8", xml_declaration=False)


if __name__ == '__main__':
    insert_node(annotation_path="E:/Data/underwater/train/box/",
                save_path="E:/Data/underwater/train/box/")
