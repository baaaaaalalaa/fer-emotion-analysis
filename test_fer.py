import cv2
from fer.fer import FER
import pandas as pd
import matplotlib.pyplot as plt


def image_detection(img_name):
    img = cv2.imread(img_name)
    if img is None:
        print(f"无法读取图片: {img_name}")
        return None

    detector = FER(mtcnn=True)
    results = detector.detect_emotions(img)

    if not results:
        print("没有检测到人脸")
        return None

    # 新增：筛选逻辑
    valid_faces = []
    for face in results:
        box = face['box']
        w = box[2]
        h = box[3]

        # 筛选条件：宽和高都必须大于 100 像素
        if w > 100 and h > 100:
            valid_faces.append(face)
        else:
            # 如果你不想看提示，可以把这行注释掉
            print(f"  [已过滤] 发现一个过小的疑似误检框: 宽{w}, 高{h}")

    if not valid_faces:
        print("没有检测到有效的人脸（全被过滤了）")
        return None

    # 遍历筛选后的有效人脸
    print(f"\n--- 图片: {img_name} ---")
    for face in valid_faces:
        box = face['box']
        emotions = face['emotions']

        # 找出得分最高的情绪
        top_emotion = max(emotions, key=emotions.get)
        top_score = emotions[top_emotion]

        print(f"  人脸位置: {box}")
        print(f"  主要情绪: {top_emotion} ({top_score:.2f})")

    return valid_faces


def analyze_video(video_path, step=5):
    cap = cv2.VideoCapture(video_path)
    detector = FER(mtcnn=True)
    emotions_data = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 隔帧采样，提升速度
        if frame_count % step == 0:
            results = detector.detect_emotions(frame)
            if results:
                # 假设只取画面中分数最高/最大的那张脸
                face = max(results, key=lambda x: x['box'][2] * x['box'][3])
                emotions = face['emotions']
                emotions['frame'] = frame_count
                emotions_data.append(emotions)

        frame_count += 1

    cap.release()

    # 转为 DataFrame 并画图
    df = pd.DataFrame(emotions_data)
    if not df.empty:
        df.set_index('frame', inplace=True)
        df.plot(title="Video Emotion Analysis")
        plt.show()
    else:
        print("未检测到有效人脸")


def main():
    for i in range(1, 4):
        # 这里用的是相对路径，请确保运行目录下有 Image 文件夹
        image_detection(f"Image/face{i}.jpg")
    analyze_video("Image/video1.mp4")


if __name__ == "__main__":
    main()