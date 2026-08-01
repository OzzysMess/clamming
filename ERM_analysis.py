# This script looks at a picture of a partial ERM plate and computes 
# total mass, and rotational ineteria about axis of rotation. Blue pixel 
# =  axis, black pixel = metal, white pixel = void. It is assumed the 
# void is filled when in use, it sheared off when taking plate off.

from pathlib import Path
from PIL import Image

image_path = Path(__file__).resolve().parent / "Data" / "ERM.png"
image = Image.open(image_path)
pixels = list(image.getdata())

opaque_pixels = [pixel for pixel in pixels if pixel[3] == 255]
black_pixels = sum(1 for pixel in opaque_pixels if pixel[:3] == (0, 0, 0))
white_pixels = sum(1 for pixel in opaque_pixels if pixel[:3] == (255, 255, 255))

print(f"black_pixels: {black_pixels}")
print(f"white_pixels: {white_pixels}")


#plate_mass = 0.43g, wiith voids

pixel_density = 0.43 / black_pixels
total_mass = pixel_density * (black_pixels + white_pixels)
print(f"total_mass: {total_mass} g")

pixel_length = 11.05 / 294  # mm/pixel, mass width/pixel width of mass

pivot = None
for y in range(image.height):
    for x in range(image.width):
        r, g, b, a = image.getpixel((x, y))
        if a == 255 and (r, g, b) == (0, 0, 255):
            pivot = (x, y)
            break
    if pivot is not None:
        break

if pivot is None:
    pivot = (image.width // 2, image.height // 2)
    print("No blue pivot pixel found; using image center as the pivot.")

pivot_x, pivot_y = pivot
moment_of_inertia_g_mm2 = 0.0
for y in range(image.height):
    for x in range(image.width):
        r, g, b, a = image.getpixel((x, y))
        if a != 255:
            continue

        dx = (x - pivot_x) * pixel_length
        dy = (y - pivot_y) * pixel_length
        radius_squared = dx * dx + dy * dy
        moment_of_inertia_g_mm2 += (0.43 / black_pixels) * radius_squared

moment_of_inertia_kg_m2 = moment_of_inertia_g_mm2 * 1e-9
print(f"pivot_pixel: {pivot}")
print(f"moment_of_inertia: {moment_of_inertia_g_mm2:.6f} g*mm^2")
print(f"moment_of_inertia: {moment_of_inertia_kg_m2:.6e} kg*m^2")
