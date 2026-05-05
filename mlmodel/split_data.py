import os, shutil, random

src = r"C:\Users\Meghana\Downloads\archive (2)\ALL_IDB Dataset"
dst = r"C:\Users\Meghana\project\ML model\dataset"

for cls in ['L1', 'L2', 'L3']:
    images = os.listdir(os.path.join(src, cls))
    random.shuffle(images)
    split = int(0.8 * len(images))
    
    for phase, imgs in [('train', images[:split]), ('val', images[split:])]:
        out_dir = os.path.join(dst, phase, cls)
        os.makedirs(out_dir, exist_ok=True)
        for img in imgs:
            shutil.copy(os.path.join(src, cls, img), os.path.join(out_dir, img))

print("Done! Dataset split complete.")