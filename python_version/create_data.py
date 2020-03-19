from PIL import Image, ImageDraw, ImageFont

# A list of all fonts composed from:
# 1. Font Name
# 2. (x, y) locations
# 3. Font Size

text_font = [
    ('CALIST.TTF', (-1, -8), 35),
    ('CALIST.TTF', (0, -5), 30),
    ('CALISTBI.TTF', (-1, -8), 35),
    ('CALISTBI.TTF', (0, -5), 30),
    ('CALISTI.TTF', (-1, -8), 35),
    ('CALISTI.TTF', (0, -5), 30),
    ('david.ttf', (-1, -5), 43),
    ('david.ttf', (0, -4), 36)
]
for i, (font, pos, font_size) in enumerate(text_font):
    for j in range(0, 10):
        img = Image.new('L', (18, 30), color=255)

        fnt = ImageFont.truetype('../assets/' + str(font), font_size)
        d = ImageDraw.Draw(img)
        d.text(pos, str(j), font=fnt, fill=100)

        img.save('./new_data/' + str(i) + str(j) + '.png')

print("Done")