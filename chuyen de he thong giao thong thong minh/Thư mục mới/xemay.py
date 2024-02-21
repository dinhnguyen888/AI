'''
sử dụng một khối try-except để xử lý các loại lỗi có thể xảy ra
sử dụng các biến screenCnt, result, và new_img để lưu trữ và sử dụng thông tin trong quá trình xử lý hình ảnh
cung cấp thông báo lỗi cụ thể khi có lỗi xảy ra, trong khi Code 1 chỉ in lỗi không xác định
'''
import cv2
from matplotlib import pyplot as plt
import imutils

# Đọc hình ảnh từ tệp bienso2.jpg
try:
    image = cv2.imread("bienso2.jpg")

    # Thay đổi kích thước của ảnh
    image = imutils.resize(image, width=300)

    # Chuyển đổi từ màu BGR sang màu RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hiển thị hình ảnh đã được xử lý
    plt.imshow(img)

    # Chuyển sang màu xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bộ lọc Bilateral để giảm nhiễu
    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

    # Canny Edge Detection để phát hiện biên cạnh
    edged = cv2.Canny(gray_image, 30, 200)

    # Hiển thị hình ảnh xám
    plt.imshow(edged, cmap="gray")

    # Tìm các đường viền trong hình ảnh biên cạnh
    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp danh sách các đường viền dựa trên diện tích
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    # Tạo bản sao của hình ảnh gốc để vẽ đường viền lên đó
    image2 = image.copy()

    # Chuyển đổi sang màu RGB
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Vẽ các đường viền lên hình ảnh image2
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 1)

    # Hiển thị hình ảnh sau khi đã vẽ đường viền
    plt.imshow(image2)

    screenCnt = None

    # Vòng lặp duyệt qua danh sách đường viền
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    # Nếu tìm thấy hình chữ nhật, vẽ đường viền lên hình ảnh gốc
    if screenCnt is not None:
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)

    # Chuyển đổi sang màu RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hiển thị hình ảnh cuối cùng
    plt.imshow(image)
    plt.axis("off")
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")
