from matplotlib import pyplot as plt
import torchvision.transforms as tf
import torchvision.transforms.functional as F

def viz_torch(img, mask):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(tf.functional.to_pil_image(img))
    plt.subplot(122)
    plt.imshow(tf.functional.to_pil_image(mask.float()))
    plt.show()


if __name__ == "__main__":
    pass
# # test()
# pppppppppp()
# main()
# viz_model()
# show_mask_true_and_predict()
