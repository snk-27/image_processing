import streamlit as st
import cv2
import numpy as np

# Pre-trained face detector dataset
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# COUNT FACE FROM IMAGE
def detect_faces_in_image(image, scaleFactor, minNeighbors):

    # Check if user uploads a picture successfully
    if image is not None:
        st.success("Image Uploaded Successfully")
    else:
        st.warning("No image uploaded.")
        return None, 0

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image with rectangles around the faces in rgb/correct color channel
    resized_image = cv2.resize(image_rgb, (600, 600)) 
    return resized_image, len(faces)


# COUNT FACE FROM LIVE WEBCAM
def detect_faces_in_webcam(scaleFactor, minNeighbors):

    # Open the webcam
    webcam = cv2.VideoCapture(0)

    # Check if the webcam is open or not
    if not webcam.isOpened():
        st.error("Error: Unable to open webcam.")
        return
    
    # Create a streamlit image element to display the webcam feed
    webcam_image = st.empty()
    
    while True:
        # Capture the cam frame-by-frame
        ret, frame = webcam.read()

        # Check if frame is valid
        if not ret:
            st.error("Error: Unable to read frame from webcam.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

        # Draw rectangles around the detected faces and count them
        face_count = 0
        for (x, y, w, h) in faces:
            face_count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {face_count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with rectangles around the faces and face count
        cv2.putText(frame, f"Total Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the updated frame in the streamlit app
        webcam_image.image(frame, channels="BGR", use_column_width=True)


# SIDEBAR
st.sidebar.title("-Face Counting Options-")
option = st.sidebar.selectbox("Select Option", ("None", "From Image", "From Live Webcam"))
stop_flag = False # Stop flag for webcam feed/stop the webcam

# Display content based on the selected option
if option == "None":
    st.title("Welcome to our Face Counting App")
    st.caption("Please select an option from the sidebar.")
    st.write("How to Use :")
    st.markdown("1. **Select an option** from the sidebar.")
    st.image("C:/Users/User/Desktop/image_processing-main/img/homeImg1.png", caption="Step 1: Select an option", use_column_width=True)
    st.markdown("2. **Upload an image** if you choose the 'From Image' option.")
    st.image("C:/Users/User/Desktop/image_processing-main/img/homeImg2.png", caption="Step 2: Upload an image", use_column_width=True)
    st.markdown("3. **Click 'Detect Faces'** to detect faces in the uploaded image.")
    st.image("C:/Users/User/Desktop/image_processing-main/img/homeImg3.png", caption="Step 3: Click 'Detect Faces'", use_column_width=True)
    st.image("C:/Users/User/Desktop/image_processing-main/img/homeImg4.png", caption="Result", use_column_width=True)
    st.markdown("4. **Click 'From Live Webcam'** to detect faces based on the live webcam.")
    st.image("C:/Users/User/Desktop/image_processing-main/img/homeImg5.png", caption="Step 4: Click 'Live Webcam'", use_column_width=True)
    st.markdown("5. **Click 'Start Webcam'** to start the live counting.")
    st.image("C:/Users/User/Desktop/image_processing-main/img/homeImg6.png", caption="Step 5: Click 'Start Webcam'", use_column_width=True)
    st.image("C:/Users/User/Desktop/image_processing-main/img/homeImg6.1.png", caption="Result", use_column_width=True)
    

elif option == "From Image":
    st.title("Counting Face from image")

    # Update the scaleFactor slider
    scaleFactor = st.sidebar.slider("Scale Factor", min_value=1.01, max_value=1.5, value=1.1, step=0.01, key="image_scaleFactor")
    # Update the minNeighbors slider
    minNeighbors = st.sidebar.slider("Minimum Neighbors", min_value=1, max_value=10, value=5, step=1, key="image_minNeighbors")
    uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_image is not None:
        try:
            # Convert uploaded image to NumPy array to get the correct format required in the function detect_faces_in_image
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
              
            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Detect and count faces in the uploaded image
            resized_image, num_faces = detect_faces_in_image(image, scaleFactor, minNeighbors)
            st.image(resized_image, caption=f'Number of Faces Detected: {num_faces}', use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
            
            
elif option == "From Live Webcam":
    st.title("Live Webcam Feed")

    # Update the scaleFactor slider
    scaleFactor = st.sidebar.slider("Scale Factor", min_value=1.01, max_value=1.5, value=1.1, step=0.01, key="webcam_scaleFactor")
    # Update the minNeighbors slider
    minNeighbors = st.sidebar.slider("Minimum Neighbors", min_value=1, max_value=10, value=5, step=1, key="webcam_minNeighbors")
    
    # Button to start/stop webcam feed
    stop_button = st.button("Stop Webcam")
    start_button = st.button("Start Webcam")
    if stop_button: 
        stop_flag = True #determine whether webcam is stopped
    elif start_button:
        stop_flag = False #determine whether webcam is opened
        detect_faces_in_webcam(scaleFactor, minNeighbors)
