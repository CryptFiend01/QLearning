from PIL import Image

img = Image.open('car.png')
w, h = img.size
color = img.getpixel((0, 0))

for x in range(w):
    for y in range(h):
        c = img.getpixel((x, y))
        if c == color:
            new_color = c[:-1] + (0,)
            img.putpixel((x, y), new_color)

img.save('car1.png', 'PNG')