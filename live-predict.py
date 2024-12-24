from PIL import Image

from ultralytics import YOLO

model = YOLO('YOLO9000.pt')

results = model(['./predict/randompic/randompic1.jpg','./predict/randompic/randompic2.jpg','./predict/randompic/randompic3.jpg'])

for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f"results{i}.jpg")
    