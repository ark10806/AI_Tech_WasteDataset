from PIL import Image, ImageDraw

def pil_draw_rect(image: Image, pt1: tuple, pt2: tuple):
    draw = ImageDraw.Draw(image)
    draw.rectangle((pt1, pt2), outline=(0,0,255), width=3)

    return image
