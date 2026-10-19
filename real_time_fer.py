import cv2
from fer.fer import FER


def main():
    # 1. 打开默认摄像头 (0 表示电脑默认摄像头，如果有多个可以改成 1, 2 等)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头，请检查是否被其他程序占用。")
        return

    # 2. 初始化 FER 检测器
    # 建议使用 mtcnn=False (默认的 Haar 级联)，因为 MTCNN 在普通 CPU 上处理视频帧会非常卡顿
    detector = FER(mtcnn=True)

    print("摄像头已启动，按 'q' 键退出...")

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break

        # 为了提升实时性，可以将画面缩小一半再进行检测
        # 检测完再把坐标乘 2 还原，这里为了代码简单先直接检测
        results = detector.detect_emotions(frame)

        # 遍历画面中的每一张脸
        for face in results:
            # 获取人脸的坐标框 [x, y, 宽, 高]
            (x, y, w, h) = face['box']

            # 过滤掉面积太小的误检框 (宽高都大于 80 像素才认为是真的人脸)
            if w > 80 and h > 80:
                # 获取情绪字典
                emotions = face['emotions']
                # 找出得分最高的情绪
                top_emotion = max(emotions, key=emotions.get)
                top_score = emotions[top_emotion]

                # 在画面上画一个绿色的矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 准备要显示的文字，比如 "happy: 0.95"
                label = f"{top_emotion}: {top_score:.2f}"

                # 在矩形框的上方写上文字
                # 参数分别是：图像, 文字, 坐标, 字体, 字体大小, 颜色(BGR), 线条粗细
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示画面
        cv2.imshow("Real-time Emotion Detection", frame)

        # 监听键盘按键，如果按下 'q' 键则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()