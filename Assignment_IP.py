import streamlit as st
import cv2

# Function to detect and count faces in an image
def detect_faces_in_image(image_path):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=10)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the image with rectangles around the faces
    resized_image = cv2.resize(image, (600, 600)) 
    return resized_image, len(faces)

# Function to detect and count faces from live webcam
def detect_faces_in_webcam():
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect and count face from live webcam
    webcam = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = webcam.read()

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
        st.image(frame, channels="BGR")

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam
    webcam.release()
    cv2.destroyAllWindows()

# Sidebar for selecting the option
option = st.sidebar.selectbox("Select Option", ("None", "Image", "Live Webcam"))

if option == "None":
    st.title("Welcome to Face Detection App")
    st.write("Please select an option from the sidebar.")
    
elif option == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_image is not None:
        # Display the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Detect and count faces in the uploaded image
        if st.button("Detect Faces"):
            resized_image, num_faces = detect_faces_in_image(image)
            st.image(resized_image, caption=f'Faces Detected: {num_faces}', use_column_width=True)

elif option == "Live Webcam":
    # Display the live webcam feed
    st.write("Live Webcam Feed")
    detect_faces_in_webcam()
