import json
d = json.load(open(r'C:\Users\l\Desktop\fish_dataset\annotations\coco_detection_train.json', encoding='utf-8'))
print('categories:', d['categories'])
print('标注里的category_id集合:', sorted(set(a['category_id'] for a in d['annotations'])))