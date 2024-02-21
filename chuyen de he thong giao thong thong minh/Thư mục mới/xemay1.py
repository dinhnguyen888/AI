import cv2
from  matplotlib import pyplot as plt
import imutils
#dùng để đọc hình ảnh từ tệp bienso1
image = cv2.imread("bienso2.jpg")
#trả về thông tin kích thước của ảnh bao gồm chiều rộng va chiều cao số kênh màu nếu có
image.shape
#dùng để thay đổi kích thước của ảnh
image1 = imutils.resize(image, width=300)
# dùng để chuyển đổi từ màu BGR sang màu RGB
img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)


# dùng để chuyển đổi từ màu BGR sang màu RGB
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Bộ lọc Bilateral là một loại bộ lọc trong xử lý hình ảnh được sử dụng để giảm nhiễu
gray_image = cv2.bilateralFilter(gray_image, 11,17,17)
# Canny Edge Detection là một phương pháp phát hiện biên cạnh trong xử lý hình ảnh
edged = cv2.Canny(gray_image, 30, 200)
# dùng để đổi sang màu xám
plt.imshow(edged, cmap="gray")
#để tìm các đường viền trong hình ảnh biên cạnh edged
cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#Dòng này tạo một bản sao của hình ảnh gốc image để vẽ các đường viền lên đó mà không làm thay đổi hình ảnh gốc.
image3 =  image.copy()
# dòng này để đổi màu BGR sang RGB
image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
#để vẽ các đường viền (cnts) lên hình ảnh
cv2.drawContours(image3, cnts,-1,(0,255,0),1)
# dùng để hiển thị hình ảnh sau khi đã được vẽ đường viền lên đó
plt.imshow(image3)
#Dòng này sắp xếp danh sách các đường viền (cnts) dựa trên diện tích của mỗi đường viền và chọn ra đường viền lớn nhất
cnts = sorted(cnts, key = cv2.contourArea, reverse= True)[:10]
#Dòng này tạo một bản sao của hình ảnh gốc image để vẽ các đường viền lên đó mà không làm thay đổi hình ảnh gốc.
image2 = image.copy()
# dòng này để đổi màu BGR sang RGB
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# dùng để vẽ các đường viền  cnts lên hình ảnh image2
cv2.drawContours(image2, cnts, -1,(0,255,0),1)

screenCNt = None
#  Dòng này tắt hiển thị các trục x và y trên biểu đồ
plt.axis("off")
#vòng lặp  duyệt qua các danh sách đường viền cnts
for c in cnts:
    #Dòng này tính chu vi của đường viền c bằng cách sử dụng hàm 
    perimeter = cv2.arcLength(c, True)
    #Dòng này xác định một đa giác xấp xỉ gần nhất cho đường viền c
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    # Dòng này kiểm tra xem đa giác xấp xỉ có 4 đỉnh hay không. Nếu có đúng 4 đỉnh, đó có thể là một hình chữ nhật
    if len(approx) == 4:
        #Nếu đa giác xấp xỉ có 4 đỉnh, bạn lưu nó vào biến screenCnt
        screenCnt = approx
    #Dòng này tính toán hình chữ nhật giới hạn bao quanh đường viền c
    x,y,w,h =cv2.boundingRect(c)
    #Dòng này trích xuất hình ảnh con bên trong hình chữ nhật bằng cách cắt ra một phần của hình ảnh gốc image bằng tọa độ và kích thước đã tính toán
    new_img = image[y:y+h,x:x+w]
    
    break
#Dòng này sử dụng hàm cv2.drawContours() để vẽ đường viền (screenCnt) lên hình ảnh gốc (image)
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)
# dòng này để đổi màu BGR sang RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()