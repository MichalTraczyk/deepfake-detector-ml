import cv2
from FaceProcessor import ImageFaceProcessor

image = cv2.imread("data/Train/Fake/fake_0.jpg")
face_processor = ImageFaceProcessor((256,256))
face = face_processor.get_face(image)

# Display the resulting image
cv2.imshow('Image', face)

cv2.waitKey(0)
cv2.destroyAllWindows()