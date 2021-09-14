import xml.etree.ElementTree as ET
import os


def see_label(annotation_path, class_list=[]):
    for xml_file in os.listdir(annotation_path):
        tree = ET.parse(os.path.join(annotation_path, xml_file))
        root = tree.getroot()

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls in class_list:
                continue
            else:
                class_list.append(cls)
    print(class_list)


def count_label(annotation_path, class_count={}):
    for xml_file in os.listdir(annotation_path):
        tree = ET.parse(os.path.join(annotation_path, xml_file))
        root = tree.getroot()

        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls in class_count:
                class_count[cls] += 1
            else:
                class_count[cls] = 1
    print(class_count)


if __name__ == "__main__":
    annotation_path = "E:/Data/underwater/dataset/annotations/train/"
    see_label(annotation_path)
