import cv2
from fer.fer import FER
import pandas as pd
import matplotlib.pyplot as plt
import time


def analyze_data(data_records, total_frames):
    # (分析部分代码保持不变)
    if not data_records:
        print("\n没有收集到有效数据，无法分析。")
        return

    df = pd.DataFrame(data_records)
    df.set_index('time', inplace=True)

    print("\n正在生成折线图...")
    plot_df = df.loc[:, (df != 0).any(axis=0)]
    plot_df.plot(title="Real-time Emotion Analysis (Experiment Phase)", figsize=(10, 6), ylim=(-0.05, 1.05))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 40)
    print("📊 实时检测数据分析报告 (仅统计实验阶段)")
    print("=" * 40)

    detected_frames = len(df)
    detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0
    print(f"1. 全局人脸检测率: {detection_rate:.2f}% (检测到 {detected_frames} 帧 / 实验总计 {total_frames} 帧)")

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

    detector = FER(mtcnn=True)

    # 实验控制变量
    BASELINE_DURATION = 3.0  # 基线时长（秒）
    experiment_started = False
    baseline_start_time = time.time()

    data_records = []
    total_frames = 0
    experiment_start_time = 0

    print("摄像头已启动。准备开始实验...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # ---- 阶段控制逻辑 ----
        if not experiment_started:
            # 处于基线阶段
            elapsed = current_time - baseline_start_time
            remaining = BASELINE_DURATION - elapsed

            if remaining > 0:
                # 在画面上显示倒计时
                cv2.putText(frame, f"BASELINE: {remaining:.1f}s", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, "Please keep neutral", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # 基线结束，进入实验阶段
                experiment_started = True
                experiment_start_time = time.time()
                print(">>> 基线结束，实验开始！请做出目标表情。")

        else:
            # 处于实验阶段，正式开始记录数据
            total_frames += 1
            relative_time = current_time - experiment_start_time

            # 画面左上角提示实验进行中
            cv2.putText(frame, f"EXPERIMENT: {relative_time:.1f}s", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            results = detector.detect_emotions(frame)
            valid_faces = [f for f in results if f['box'][2] > 80 and f['box'][3] > 80]

            if valid_faces:
                face = max(valid_faces, key=lambda x: x['box'][2] * x['box'][3])
                (x, y, w, h) = face['box']
                emotions = face['emotions']

                # 记录数据 (注意这里的时间是相对于实验开始的)
                record = {'time': relative_time}
                record.update(emotions)
                data_records.append(record)

                top_emotion = max(emotions, key=emotions.get)
                top_score = emotions[top_emotion]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{top_emotion}: {top_score:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Real-time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 只有在实验阶段有数据时才分析
    if experiment_started and data_records:
        analyze_data(data_records, total_frames)
    else:
        print("实验未完成或没有数据。")


if __name__ == "__main__":
    main()