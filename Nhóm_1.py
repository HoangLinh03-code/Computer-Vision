import cv2
import matplotlib.pyplot as plt
from imutils import contours
import numpy as np
import imutils
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import csv
# # Đọc ảnh
# image = cv2.imread(MaDe)
# image = cv2.resize(image, (1830, 2560))
# Cắt vùng quan tâm trên ảnh

ANSWER_KEY_1= [
    "A", "C", "D", "A", "C", "C", "D", "B", "A", "C", "A", "B", "C", "D",
    "A", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C",
    "D", "A", "A", "B", "C", "D", "A", "C", "D", "B", "A", "C", "A", "B",
    "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A",
    "B", "C", "D", "A", "A", "B", "C", "D", "A", "C", "D", "B", "A", "C",
    "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C", "D",
    "A", "A", "B", "C", "D", "D", "A", "B", "C", "D", "A", "C", "D", "B",
    "A", "C", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B",
    "C", "D", "A", "A", "B", "C", "D", "A"
]

ANSWER_KEY_2 = [
    "A", "B", "C", "D", "A", "C", "D", "B", "A", "C", "A", "B", "C", "D",
    "A", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C",
    "D", "A", "A", "B", "C", "D", "A", "C", "D", "B", "A", "C", "A", "B",
    "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A",
    "B", "C", "D", "A", "A", "B", "C", "D", "A", "C", "D", "B", "A", "C",
    "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C", "D",
    "A", "A", "B", "C", "D", "D", "A", "B", "C", "D", "A", "C", "D", "B",
    "A", "C", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B",
    "C", "D", "A", "A", "B", "C", "D", "A"
]
ANSWER_KEY_3 = [
    "A", "B", "C", "D", "A", "C", "D", "B", "A", "C", "A", "B", "C", "D",
    "A", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C",
    "D", "A", "A", "B", "C", "D", "A", "C", "D", "B", "A", "C", "A", "B",
    "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A",
    "B", "C", "D", "A", "A", "B", "C", "D", "A", "C", "D", "B", "A", "C",
    "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C", "D",
    "A", "A", "B", "C", "D", "D", "A", "B", "C", "D", "A", "C", "D", "B",
    "A", "C", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B",
    "C", "D", "A", "A", "B", "C", "D", "A"
]

ANSWER_KEY_4 = [
    "A", "B", "C", "D", "A", "C", "D", "B", "A", "C", "A", "B", "C", "D",
    "A", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C",
    "D", "A", "A", "B", "C", "D", "A", "C", "D", "B", "A", "C", "A", "B",
    "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A",
    "B", "C", "D", "A", "A", "B", "C", "D", "A", "C", "D", "B", "A", "C",
    "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B", "C", "D",
    "A", "A", "B", "C", "D", "D", "A", "B", "C", "D", "A", "C", "D", "B",
    "A", "C", "A", "B", "C", "D", "A", "A", "B", "C", "D", "A", "A", "B",
    "C", "D", "A", "A", "B", "C", "D", "A"
]


def get_result_trac_nghiem(image_trac_nghiem, ANSWER_KEY):
    translate = {"A": 0, "B": 1, "C": 2, "D": 3}
    revert_translate = {0: "A", 1: "B", 2: "C", 3: "D", -1: "N"}
    image = image_trac_nghiem
    height, width, channels = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,
                                   maxValue=255,
                                   adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                   thresholdType=cv2.THRESH_BINARY_INV,
                                   blockSize=15,
                                   C=8)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)

        ar = w / float(h)
        if w >= width / 25 and h >= height / 68 and ar >= 0.93 and ar <= 1.2 and w < width / 5 and h < height / 4 :
            questionCnts.append(c)

    questionCnts = contours.sort_contours(questionCnts,method="top-to-bottom")[0]
    # print(len(questionCnts))
    select = []
    list_min_black = []
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    min_black = 1000000000
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if total <= min_black:
                min_black = total
        if (i + 4) % 20 == 0:
            list_min_black.append(min_black)
            min_black = 1000000000
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        min_black = list_min_black[int((i) / 20)]
        cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
        list_total = []
        total_max = -1
        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if total > total_max:
                total_max = total
            if total > 0:
                list_total.append((total, j))
        answer_q = [char for char in ANSWER_KEY[q]]
        list_answer = []
        list_select = ''
        for tt in list_total:
            if tt[0] > min_black * 1.5 and tt[0] > total_max * 0.7:
                list_answer.append(tt[1])
                list_select = list_select + revert_translate[tt[1]]
        for answer in answer_q:
            color = (0, 255, 0) # Green
            k = translate[answer]
            if k in list_answer:
                color = (255, 0, 0) # Red
            cv2.drawContours(image, [cnts[k]], -1, color, 3)

            x,y,w_,h_ = cv2.boundingRect(cnts[k])
            image = cv2.putText(image, answer, (x,y), fontScale=1, color=(97, 12, 159) , thickness=2, fontFace=cv2.LINE_AA)

        select.append(list_select)

    return select, image

def process_image(image_path):
    translate = {"A": 0, "B": 1, "C": 2, "D": 3}
    revert_translate = {0: "A", 1: "B", 2: "C", 3: "D", -1: "N"}
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1830, 2560))
    height, width, channels = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,
                                    maxValue=255,
                                    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                    thresholdType=cv2.THRESH_BINARY_INV,
                                    blockSize=15,
                                    C=8)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    i = 0
    crop = []
    for contour in cnts:
        if cv2.contourArea(contour) > 50000:
            crop.append(i)
        i += 1
    img_height, img_width, img_channels = image.shape
    max_weight = 1807
    max_height = 2555
    crop_mdt = (int(1300 / max_weight * img_width),#dọc trái
                int(800 / max_height * img_height),#ngang trên
                int(1700 / max_weight * img_width),#dọc phải
                int(2440 / max_height * img_height))#ngang dưới
    cropped_image = image[crop_mdt[1]:crop_mdt[3], crop_mdt[0]:crop_mdt[2]]
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    x,y,w,h= cv2.boundingRect(cnts[crop[5]])
    ci2=image[y:y+h, x:x+w]
    ci2 = cv2.cvtColor(ci2, cv2.COLOR_BGR2GRAY)
    ci2 = cv2.GaussianBlur(ci2, (5, 5), 0)
    ci2_rgb = cv2.cvtColor(ci2, cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    column_width = w // 6
    bubble_positions = [i*51 for i in range(10)]
    test_id = ""
    for i in range(6):
        column = ci2[:, (i * column_width):((i + 1) * column_width)]
        selected_number = None
        for pos in bubble_positions:
            if np.mean(column[pos:pos + 51, :]) < 200:  # Adjust the threshold as needed
                selected_number = bubble_positions.index(pos)
                break

        if selected_number is not None:
            test_id += str(selected_number)
        else:
            test_id += "?"  # Indicates an unselected number

    
    x,y,w,h= cv2.boundingRect(cnts[crop[4]])
    ci=image[y:y+h, x:x+w]
    ci = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
    ci = cv2.GaussianBlur(ci, (5, 5), 0)
    ci_rgb = cv2.cvtColor(ci, cv2.COLOR_BGR2RGB)
    
    
    _, thresh = cv2.threshold(ci, 200, 255, cv2.THRESH_BINARY)
    column_width = w // 3
    bubble_positions = [i*51 for i in range(10)]
    student_id = ""
    for i in range(3):
        column = ci[:, (i * column_width):((i + 1) * column_width)]
        selected_number = None
        for pos in bubble_positions:
            if np.mean(column[pos:pos + 51, :]) < 200:  
                selected_number = bubble_positions.index(pos)
                break

        if selected_number is not None:
            student_id += str(selected_number)
        else:
            student_id += "?"  
    img_height, img_width, img_channels = image.shape
    max_weight = 1807
    max_height = 2555
    crop_mdt1 = (int(90 / max_weight * img_width),#dọc trái
                int(800 / max_height * img_height),#ngang trên
                int(450 / max_weight * img_width),#dọc phải
                int(2440 / max_height * img_height))#ngang dưới

    cropped_image1 = image[crop_mdt1[1]:crop_mdt1[3], crop_mdt1[0]:crop_mdt1[2]]

    image_1_30 = get_result_trac_nghiem(cropped_image1,ANSWER_KEY_1[0:30])
    
    crop_mdt2 = (int(500 / max_weight * img_width),#dọc trái
                int(800 / max_height * img_height),#ngang trên
                int(850 / max_weight * img_width),#dọc phải
                int(2440 / max_height * img_height))#ngang dưới
    cropped_image2 = image[crop_mdt2[1]:crop_mdt2[3], crop_mdt2[0]:crop_mdt2[2]]
    image2 = get_result_trac_nghiem(cropped_image2,ANSWER_KEY_2[0:30])
    
    crop_mdt3 = (int(900 / max_weight * img_width),#dọc trái
            int(800 / max_height * img_height),#ngang trên
            int(1300 / max_weight * img_width),#dọc phải
            int(2440 / max_height * img_height))#ngang dưới

    cropped_image3 = image[crop_mdt3[1]:crop_mdt3[3], crop_mdt3[0]:crop_mdt3[2]]
    image3 = get_result_trac_nghiem(cropped_image3,ANSWER_KEY_3[0:30])
    
    crop_mdt4 = (int(1300 / max_weight * img_width),#dọc trái
            int(800 / max_height * img_height),#ngang trên
            int(1700 / max_weight * img_width),#dọc phải
            int(2440 / max_height * img_height))#ngang dưới
    cropped_image4 = image[crop_mdt4[1]:crop_mdt4[3], crop_mdt4[0]:crop_mdt4[2]]
    image4 = get_result_trac_nghiem(cropped_image4,ANSWER_KEY_4[0:30])

    score_1_30 = 0
    for i in range(len(image_1_30[0])):
        if image_1_30[0][i] == ANSWER_KEY_1[i]: 
            score_1_30+=1
    score_2 = 0
    for i in range(len(image2[0])):
        if image2[0][i] == ANSWER_KEY_2[i]: 
            score_2+=1
    score_3 = 0
    for i in range(len(image3[0])):
        if image3[0][i] == ANSWER_KEY_3[i]: 
            score_3+=1
    score_4 = 0
    for i in range(len(image4[0])):
        if image4[0][i] == ANSWER_KEY_4[i]: 
            score_4+=1
    score = score_1_30 + score_2 + score_3 + score_4
    return score/120*10, test_id, student_id


def save_to_csv(scores, exam_code,registration_number):
    with open('exam_results.csv', 'a', newline='\n',encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Điểm', scores])
        writer.writerow(['Số Báo Danh', registration_number])
        writer.writerow(['Mã Đề', exam_code])

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        scores, registration_number, exam_code = process_image(file_path)
        result_label.config(text=f"Điểm: {scores:.4f}\nMã Đề: {exam_code}\nSố báo danh: {registration_number}", font=("Helvetica", 16))
        save_to_csv(scores,exam_code, registration_number)
        display_image(file_path)
        display_re(file_path)


def display_image(image_path):
    image = Image.open(image_path)
    image.show()

def display_re(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (1830, 2560))
    img_height, img_width, img_channels = image.shape
    max_weight = 1807
    max_height = 2555
    crop_mdt1 = (int(90 / max_weight * img_width),#dọc trái
                int(800 / max_height * img_height),#ngang trên
                int(450 / max_weight * img_width),#dọc phải
                int(2440 / max_height * img_height))#ngang dưới

    cropped_image1 = image[crop_mdt1[1]:crop_mdt1[3], crop_mdt1[0]:crop_mdt1[2]]

    image_1 = get_result_trac_nghiem(cropped_image1,ANSWER_KEY_1[0:30])
    crop_mdt2 = (int(500 / max_weight * img_width),#dọc trái
                int(800 / max_height * img_height),#ngang trên
                int(850 / max_weight * img_width),#dọc phải
                int(2440 / max_height * img_height))#ngang dưới
    cropped_image2 = image[crop_mdt2[1]:crop_mdt2[3], crop_mdt2[0]:crop_mdt2[2]]
    image2 = get_result_trac_nghiem(cropped_image2,ANSWER_KEY_2[0:30])
    
    crop_mdt3 = (int(900 / max_weight * img_width),#dọc trái
            int(800 / max_height * img_height),#ngang trên
            int(1300 / max_weight * img_width),#dọc phải
            int(2440 / max_height * img_height))#ngang dưới

    cropped_image3 = image[crop_mdt3[1]:crop_mdt3[3], crop_mdt3[0]:crop_mdt3[2]]
    image3 = get_result_trac_nghiem(cropped_image3,ANSWER_KEY_3[0:30])
    
    crop_mdt4 = (int(1300 / max_weight * img_width),#dọc trái
            int(800 / max_height * img_height),#ngang trên
            int(1700 / max_weight * img_width),#dọc phải
            int(2440 / max_height * img_height))#ngang dưới
    cropped_image4 = image[crop_mdt4[1]:crop_mdt4[3], crop_mdt4[0]:crop_mdt4[2]]
    image4 = get_result_trac_nghiem(cropped_image4,ANSWER_KEY_4[0:30])
    width = 405
    height = 1643
    # Đảm bảo rằng kích thước của các ảnh là giống nhau
    image1 = cv2.resize(image_1[1], (width, height))
    image2 = cv2.resize(image2[1], (width, height))
    image3 = cv2.resize(image3[1], (width, height))
    image4 = cv2.resize(image4[1], (width, height))

    # Tạo hình ảnh ghép
    merged_image = np.zeros((2 * height, 2 * width, 3), dtype=np.uint8)

    # Gán ảnh vào vị trí tương ứng
    merged_image[:height, :width] = image1
    merged_image[:height, width:] = image2
    merged_image[height:, :width] = image3
    merged_image[height:, width:] = image4

    merged_image = cv2.hconcat([image1, image2, image3, image4])
    plt.imshow(merged_image)
    plt.axis('off')  # Tắt trục tọa độ
    plt.show()

root = tk.Tk()
root.title("Phần Mềm Chấm Thi Trung Học Phổ Thông Quốc Gia")
root.geometry("500x500")
open_button = tk.Button(root, text="Mở Ảnh", command= open_image, bg="blue", fg="white", 
                        width= 30, height= 8,font=("Helvetica", 16),takefocus=1)
open_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()