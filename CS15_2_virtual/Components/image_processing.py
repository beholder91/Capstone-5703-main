
class Ensure3Channels:
    def __call__(self, img):
        if img.shape[0] == 1:  # grayscale
            return img.repeat(3, 1, 1)
        elif img.shape[0] == 4:  # RGBA
            return img[:3, :, :]
        elif img.shape[0] == 3:  # RGB
            return img
        else:
            print(f"Unexpected number of channels: {img.shape[0]}")
            raise ValueError(f"Unexpected number of channels: {img.shape[0]}")

class ConvertToRGBA:
    def __call__(self, img):
        if img.mode == 'P':
            return img.convert('RGBA')
        return img
