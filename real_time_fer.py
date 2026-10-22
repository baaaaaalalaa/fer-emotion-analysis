import cv2
from fer.fer import FER
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


def analyze_data(data_records, total_frames, image_name):
    """分析单个试次（单张刺激图片）的数据并画图"""
    if not data_records:
        print("\n没有收集到有效数据，无法分析。")
        return

    df = pd.DataFrame(data_records)
    df.set_index('time', inplace=True)

    # 画折线图
    plot_df = df.loc[:, (df != 0).any(axis=0)]
    plot_df.plot(title=f"Emotion Analysis - Stimulus: {image_name}", figsize=(10, 6), ylim=(-0.05, 1.05))
    plt.xlabel("Time since stimulus (seconds)")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()  # 阻塞显示，关掉图后才会继续下一个循环

    print(f"\n" + "=" * 40)
    print(f"📊 数据分析报告 - 刺激图片: {image_name}")
    print("=" * 40)

    detected_frames = len(df)
    detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0
    print(f"1. 全局人脸检测率: {detection_rate:.2f}% (检测到 {detected_frames} 帧 / 总计 {total_frames} 帧)")

    emotion_cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for col in emotion_cols:
        if col not in df.columns:
            df[col] = 0.0

    print("\n2. 每种情绪的概率统计:")
    avg_probs = df[emotion_cols].mean()
    max_probs = df[emotion_cols].max()
    for emo in emotion_cols:
        print(f"   - {emo.capitalize():<9}: 平均概率 {avg_probs[emo]:.3f} | 最大概率 {max_probs[emo]:.3f}")

    df['dominant_emotion'] = df[emotion_cols].idxmax(axis=1)
    dominant_counts = df['dominant_emotion'].value_counts()
    top_3_ratios = (dominant_counts / detected_frames * 100).head(3)

    print("\n3. 前三名情绪预测帧比率:")
    for emo, ratio in top_3_ratios.items():
        print(f"   - {emo.capitalize():<9}: {ratio:.2f}%")

    print("\n4. 前三名情绪的峰值时间与持续时间:")
    for emo in top_3_ratios.index:
        peak_time = df[emo].idxmax()
        peak_val = df[emo].max()
        avg_interval = (df.index.max() - df.index.min()) / len(df) if len(df) > 1 else 0
        duration = len(df[df[emo] > 0.5]) * avg_interval
        print(
            f"   - {emo.capitalize():<9}: 峰值在 {peak_time:.2f}s (得分 {peak_val:.2f}) | 持续时间(>0.5) 约 {duration:.2f}s")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("正在加载 FER 模型，请稍候...")
    detector = FER(mtcnn=True)

    # 实验参数设置
    BASELINE_DURATION = 3.0  # 基线时长（秒）
    EXPERIMENT_DURATION = 10.0  # 观看图片/记录数据的时长（秒）

    # 假设你的刺激图片在 stimuli 文件夹下，命名为 img1.jpg 到 img7.jpg
    # 如果你在 D:\projects\test 运行，确保有 D:\projects\test\stimuli\ 文件夹
    stimuli_dir = "Image"

    # 确保刺激图片文件夹存在
    if not os.path.exists(stimuli_dir):
        print(f"警告：没有找到文件夹 '{stimuli_dir}'，请确保刺激图片存放正确。")

    for i in range(1, 8):
        img_name = f"img{i}.jpg"
        img_path = os.path.join(stimuli_dir, img_name)

        # 读取刺激图片
        stimulus_img = cv2.imread(img_path)
        if stimulus_img is None:
            print(f"\n[跳过] 找不到刺激图片: {img_path}")
            continue

        print(f"\n" + "*" * 50)
        print(f"开始第 {i} 轮实验，准备展示: {img_name}")
        print(f"*" * 50)

        # ---- 重置这一轮的数据 ----
        data_records = []
        total_frames = 0

        # ---- 阶段一：基线（只看摄像头，不看图片）----
        baseline_start_time = time.time()
        print(f"-> 请保持平静 ({BASELINE_DURATION}秒基线)...")

        while True:
            ret, frame = cap.read()
            if not ret: break

            elapsed = time.time() - baseline_start_time
            remaining = BASELINE_DURATION - elapsed

            if remaining <= 0:
                break  # 基线结束

            cv2.putText(frame, f"BASELINE: {remaining:.1f}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(frame, "Keep neutral, relax.", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("User Camera", frame)
            cv2.waitKey(1)

        # ---- 阶段二：刺激与数据记录 ----
        print(f"-> 请观看图片！开始记录数据 ({EXPERIMENT_DURATION}秒)...")
        experiment_start_time = time.time()

        # 提前设置好刺激图片的窗口，可以调一下大小适应屏幕
        # cv2.namedWindow("Stimulus", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Stimulus", 800, 600)

        while True:
            ret, frame = cap.read()
            if not ret: break

            relative_time = time.time() - experiment_start_time

            if relative_time > EXPERIMENT_DURATION:
                break  # 实验阶段结束

            total_frames += 1

            # 实时显示摄像头画面（可选：如果你不想让用户看到自己，可以注释掉 imshow frame）
            cv2.putText(frame, f"EXPERIMENT: {relative_time:.1f}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # 进行情绪识别
            results = detector.detect_emotions(frame)
            valid_faces = [f for f in results if f['box'][2] > 80 and f['box'][3] > 80]

            if valid_faces:
                face = max(valid_faces, key=lambda x: x['box'][2] * x['box'][3])
                (x, y, w, h) = face['box']
                emotions = face['emotions']

                # 记录
                record = {'time': relative_time}
                record.update(emotions)
                data_records.append(record)

                # 绘制
                top_emotion = max(emotions, key=emotions.get)
                top_score = emotions[top_emotion]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{top_emotion}: {top_score:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 显示摄像头画面
            cv2.imshow("User Camera", frame)

            # 显示刺激图片 (在另一个窗口弹出)
            cv2.imshow("Stimulus", stimulus_img)

            cv2.waitKey(1)

        # 关闭刺激图片的窗口，准备下一轮
        cv2.destroyWindow("Stimulus")

        # ---- 阶段三：分析当前试次数据 ----
        if data_records:
            analyze_data(data_records, total_frames, img_name)
            print("\n关闭图表后，将进入下一张图片的准备阶段...")
        else:
            print("本轮没有收集到数据。")

    cap.release()
    cv2.destroyAllWindows()
    print("\n所有 7 轮实验结束！")


if __name__ == "__main__":
    main()