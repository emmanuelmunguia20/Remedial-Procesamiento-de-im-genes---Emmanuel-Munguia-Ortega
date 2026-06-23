import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


img1 = cv2.imread('input1.jpg')
img2 = cv2.imread('input2.jpg')

if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

line_thickness = 5
height = img1.shape[0]
line = np.full((height, line_thickness, 3), (0, 0, 255), dtype=np.uint8)

side_by_side = np.hstack((img1, line, img2))

side_by_side_rgb = cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.imshow(side_by_side_rgb)
plt.title('input1 input2')
plt.axis('off')
plt.show()


added = cv2.add(img1, img2)
added_rgb = cv2.cvtColor(added, cv2.COLOR_BGR2RGB)

plt.imshow(added_rgb)
plt.title('Addition (cv2.add)')
plt.axis('off')
plt.show()



weighted = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
weighted_rgb = cv2.cvtColor(weighted, cv2.COLOR_BGR2RGB)

plt.imshow(weighted_rgb)
plt.title('Weighted Addition (cv2.addWeighted)')
plt.axis('off')
plt.show()





subtracted = cv2.subtract(img1, img2)
subtracted_rgb = cv2.cvtColor(subtracted, cv2.COLOR_BGR2RGB)

plt.imshow(subtracted_rgb)
plt.title('Subtraction (cv2.subtract)')
plt.axis('off')
plt.show()




and_img = cv2.bitwise_and(img1, img2)
and_img_rgb = cv2.cvtColor(and_img, cv2.COLOR_BGR2RGB)

plt.imshow(and_img_rgb)
plt.title('Bitwise AND')
plt.axis('off')
plt.show()



or_img = cv2.bitwise_or(img1, img2)
or_img_rgb = cv2.cvtColor(or_img, cv2.COLOR_BGR2RGB)

plt.imshow(or_img_rgb)
plt.title('Bitwise OR')
plt.axis('off')
plt.show()


xor_img = cv2.bitwise_xor(img1, img2)
xor_img_rgb = cv2.cvtColor(xor_img, cv2.COLOR_BGR2RGB)

plt.imshow(xor_img_rgb)
plt.title('Bitwise XOR')
plt.axis('off')
plt.show()


not_img = cv2.bitwise_not(img1)
not_img_rgb = cv2.cvtColor(not_img, cv2.COLOR_BGR2RGB)

plt.imshow(not_img_rgb)
plt.title('Bitwise NOT (Image 1)')
plt.axis('off')
plt.show()



carpeta_resultados = "imagenes_modificadas"
os.makedirs(carpeta_resultados, exist_ok=True)

cv2.imwrite(os.path.join(carpeta_resultados, "01_imagenes_entrada.jpg"), side_by_side)
cv2.imwrite(os.path.join(carpeta_resultados, "02_suma.jpg"), added)
cv2.imwrite(os.path.join(carpeta_resultados, "03_suma_ponderada.jpg"), weighted)
cv2.imwrite(os.path.join(carpeta_resultados, "04_resta.jpg"), subtracted)
cv2.imwrite(os.path.join(carpeta_resultados, "05_and.jpg"), and_img)
cv2.imwrite(os.path.join(carpeta_resultados, "06_or.jpg"), or_img)
cv2.imwrite(os.path.join(carpeta_resultados, "07_xor.jpg"), xor_img)
cv2.imwrite(os.path.join(carpeta_resultados, "08_not.jpg"), not_img)



