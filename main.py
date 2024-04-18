import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'yolov5\runs\train\exp\weights\best.pt')
dense_model = load_model(r'DenseNet201.h5')

def draw_boxes(image, boxes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.patch.set_facecolor('black')
    for box in boxes:
        xmin, ymin, xmax, ymax, confidence, class_id = box.tolist()
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin - 20, f"Confidence: {confidence:.2f}", fontsize=8, color='r', verticalalignment='top')

    ax.axis('off')
    fig.canvas.draw()
    annotated_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    annotated_img = annotated_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return annotated_img

def preprocess_image(image_array, target_size):
    img = Image.fromarray(image_array)
    img = img.convert("RGB")  # Convert to RGB format
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize pixel values to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_and_display(image_array, model, target_size):
    # Preprocess the image
    processed_image = preprocess_image(image_array, target_size)
    
    # Generate prediction
    prediction = model.predict(processed_image)
    
    # Get the predicted class index and its corresponding label
    predicted_class_index = np.argmax(prediction)
    class1=['Class 0','Class 1','Class 2','Class 3','Class 4']
    class_labels = ['No pathological features', 'Doubtful narrowing of joint space and possible osteophytic lipping', 'Definite osteophytes and possible narrowing of joint space', 'Moderate multiple osteophytes', 'Large osteophytes']  # Update with your actual class labels
    predicted_label = class_labels[predicted_class_index]
    predicted=class1[predicted_class_index]
    # Plot the image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array.astype(np.uint8))
    plt.title('Input Image')
    plt.axis('off')

    # Display the predicted label as text
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f'Condition: {predicted}', fontsize=20, color='red', weight='bold', ha='center', va='center')
    plt.axis('off')  # Turn off axis
    
    plt.show()
    # Plot the cropped image and its predicted label
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_array, caption='Knee Image', use_column_width=True)
    with col2:
        st.write(f"Predicted : {predicted_label}")
        st.pyplot(plt)

    # Load the image for visualization
    


def main():
    st.title("Knee Osteoarthritis Severity Grading")
    uploaded_file = st.file_uploader("Upload X-ray image of knee", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        original_image = np.array(Image.open(uploaded_file))
        
        results = yolo_model(original_image)
        annotated_img = draw_boxes(original_image, results.xyxy[0])
        #original_image = original_image.resize((224, 224))
        st.subheader("Original Image")
        st.image(original_image, caption='Uploaded X-ray image', use_column_width=True)
        #annotated_img = annotated_img.resize((224, 224))
        st.subheader("Annotated Image")
        st.image(annotated_img, caption="Annotated Image", use_column_width=True)

        # Cropped images
        st.subheader("Cropped Images")
        cropped_images = []
        target_size = (224, 224)  # Change to your desired size
        
        for box in results.xyxy[0]:
            xmin, ymin, xmax, ymax, _, _ = box.tolist()
            cropped_img = original_image[int(ymin):int(ymax), int(xmin):int(xmax)]
            cropped_img = Image.fromarray(cropped_img)
            cropped_img = cropped_img.resize(target_size)
            cropped_images.append(np.array(cropped_img))

        col1, col2 = st.columns(2)
        for idx, cropped_img in enumerate(cropped_images):
            with col1 if idx % 2 == 0 else col2:

                st.image(cropped_img, caption=f"Cropped Image {idx+1}", use_column_width=True)
        if st.button("Predict"):
            st.subheader("Predicted Outputs")
            for idx, cropped_img in enumerate(cropped_images):
                prediction = predict_and_display(cropped_img, dense_model, target_size)
                





if __name__ == "__main__":
    main()
