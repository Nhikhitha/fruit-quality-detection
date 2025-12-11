import os
img_dir='datasets/valid'
lbl_dir='datasets/valid'
imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))])
lbls = sorted([f for f in os.listdir(lbl_dir) if f.endswith('.txt')])
print('images', len(imgs),'labels', len(lbls))
