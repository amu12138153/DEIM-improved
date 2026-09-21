import json

def remap_categories(ann_path, out_path):
    with open(ann_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    # 按原id排序，映射为 0, 1, ...
    old_ids = sorted(c['id'] for c in d['categories'])
    mapping = {old: new for new, old in enumerate(old_ids)}
    for c in d['categories']:
        c['id'] = mapping[c['id']]
    for a in d['annotations']:
        a['category_id'] = mapping[a['category_id']]
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False)
    print(f'{out_path}  映射关系: {mapping}')

remap_categories(r'C:\Users\l\Desktop\fish_dataset622\train\coco_detection_train.json',
                 r'C:\Users\l\Desktop\fish_dataset622\train\coco_detection_train0.json')
remap_categories(r'C:\Users\l\Desktop\fish_dataset622\val\coco_detection_val.json',
                 r'C:\Users\l\Desktop\fish_dataset622\val\coco_detection_val0.json')
# remap_categories(r'C:\Users\l\Desktop\fish_dataset622\test\coco_detection_test.json',
#                  r'C:\Users\l\Desktop\fish_dataset622\test\coco_detection_test0.json')