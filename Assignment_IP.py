import streamlit as st
import cv2
import numpy as np

# Function to detect and count faces in an image
def detect_faces_in_image(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=10)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image with rectangles around the faces in rgb/correct colour channel
    resized_image = cv2.resize(image_rgb, (600, 600)) 
    return resized_image, len(faces)

# Function to detect and count faces from live webcam
def detect_faces_in_webcam():
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the webcam
    webcam = cv2.VideoCapture(1)

    #Check if the webcam is open or not
    if not webcam.isOpened():
        st.error("Error: Unable to open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = webcam.read()

        # Check if frame is valid
        if not ret:
            st.error("Error: Unable to read frame from webcam.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        # Draw rectangles around the detected faces and count them
        face_count = 0
        for (x, y, w, h) in faces:
            face_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {face_count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with rectangles around the faces and face count
        cv2.putText(frame, f"Total Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        st.image(frame, channels="BGR", use_column_width=True)

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    webcam.release()
    cv2.destroyAllWindows()

# Sidebar dropdown list for selecting the option
option = st.sidebar.selectbox("Select Option", ("None", "Image", "Live Webcam"))

# Display content based on the selected option
if option == "None":
    st.title("Welcome to Face Detection App")
    st.write("Please select an option from the sidebar.")
elif option == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_image is not None:
        try:
            # Convert uploaded image to NumPy array to get the correct format required in the function detect_faces_in_image
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Detect and count faces in the uploaded image
            if st.button("Detect Faces"):
                resized_image, num_faces = detect_faces_in_image(image)
                st.image(resized_image, caption=f'Number of Faces Detected: {num_faces}', use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
elif option == "Live Webcam":
    st.write("Live Webcam Feed")
    detect_faces_in_webcam()

