import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model/video.h5')

# Define the emotion labels
emotion_labels = ['Fear', 'neutral', 'sad', 'anger', 'surprise', 'disgust', 'Happy']

# Load the haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Read each frame from the video capture
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        roi = gray[y:y + h, x:x + w]
        
        # Resize the ROI to match the input size of the model
        resized_roi = cv2.resize(roi, (48, 48))
        
        # Normalize the resized ROI
        normalized_roi = resized_roi / 255.0
        
        # Reshape the normalized ROI to match the model's input shape
        reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
        
        # Perform emotion prediction
        predictions = model.predict(reshaped_roi)
        
        # Get the index of the predicted emotion
        predicted_emotion_index = np.argmax(predictions[0])
        
        # Get the corresponding emotion label
        predicted_emotion_label = emotion_labels[predicted_emotion_index]
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Display the predicted emotion label on the frame
        cv2.putText(frame, predicted_emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()
