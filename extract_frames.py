import cv2


def extract_frames(video_path, output_folder, frame_rate=1, start_time=0):
    cap = cv2.VideoCapture(video_path)

    # Set the starting frame based on the start_time
    start_frame = int(start_time * cap.get(cv2.CAP_PROP_FPS))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number >= start_frame and frame_number % frame_rate == 0:
            output_path = f"{output_folder}/v1frame{frame_number}.jpg"
            cv2.imwrite(output_path, frame)

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()


video_path = "/home/decamargo/Downloads/recording1.mp4"
output_folder = "./data/pov"
frame_rate = 5  # Extract one frame per 10 frames
start_time = 15  # Start extracting frames from the 5th second
extract_frames(video_path, output_folder, frame_rate, start_time)
