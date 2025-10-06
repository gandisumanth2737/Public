import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Dummy main function — replace with your actual logic
def main(image, a_temp, b_temp, i_temp, j_temp):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Initialize variables to store the longest horizontal and vertical lines
    longest_horizontal = None
    longest_vertical = None
    max_h_length = 0
    max_v_length = 0

    # Image dimensions
    height, width = image.shape[:2]

    # Find the longest horizontal and vertical lines starting from bottom-left
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)

        # Check for horizontal lines near the bottom
        if abs(y1 - y2) < 5 and min(y1, y2) > height * 0.8:
            if length > max_h_length:
                max_h_length = length
                longest_horizontal = (x1, y1, x2, y2)

        # Check for vertical lines near the left
        if abs(x1 - x2) < 5 and min(x1, x2) < width * 0.3:
            if length > max_v_length:
                max_v_length = length
                longest_vertical = (x1, y1, x2, y2)

    # Define crop boundaries using both endpoints of the detected axis lines
    if longest_horizontal and longest_vertical:
        x_crop_start = longest_vertical[0] 
        x_crop_end = max(longest_horizontal[0], longest_horizontal[2])
        y_crop_start = min(longest_vertical[1], longest_vertical[3])
        y_crop_end = longest_horizontal[1]

       
        x_crop_start = max(0, x_crop_start)+8
        x_crop_end = min(width, x_crop_end) -5
        y_crop_start = max(0, y_crop_start)+5
        y_crop_end = min(height, y_crop_end) -5

        cropped_image = image[y_crop_start:y_crop_end, x_crop_start:x_crop_end]
        # Get image dimensions
        
        # Save original image dimensions before cropping
        original_height, original_width = image.shape[:2]

        # Convert cropped image to grayscale
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_height, cropped_width = cropped_image.shape[:2]

        # Assume 0.1 cm corresponds to a fixed number of pixels (e.g., 5 pixels)
        # This can be adjusted based on actual DPI if known
        slice_width = 1

        # Initialize lists to store coordinates
        x_coords = []
        y_coords = []

        # Sweep horizontally across the image
        for x in range(0, width, slice_width):
            # Extract vertical slice
            slice = cropped_image[:, x:x+slice_width]

            # Find the darkest pixel in the slice
            min_val = 255
            min_y = None
            for col in range(slice.shape[1]):
                for row in range(slice.shape[0]):
                    if slice[row, col] < min_val:
                        min_val = slice[row, col]
                        min_y = row

           

            norm_x = a_temp + ((x + slice_width // 2) / cropped_width * (b_temp - a_temp))

            if min_y is not None:
                absolute_y = y_crop_start + min_y
                norm_y = i_temp + ((1 - absolute_y / cropped_height) * (j_temp - i_temp))
            else:
                norm_y = None

            if norm_y is not None:
                x_coords.append(norm_x)
                y_coords.append(norm_y)

        df = pd.DataFrame({'x': x_coords, 'y': y_coords})

        # Save Excel to buffer
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        #PLOT IMAGE
        
        # Create the plot
        fig, plot_buffer = plt.subplots()
        plot_buffer.plot(df['x'], df['y'], color='purple', linewidth=0.5)
        plot_buffer.set_xlim([a_temp, b_temp])
        plot_buffer.set_ylim([i_temp, j_temp])

        # Save the plot to an image buffer
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)


    return plot_buffer, excel_buffer

# UI
st.title("Image Processing App")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # input_image = Image.open(uploaded_file).convert("RGB")
    # input_image = cv2.imread(uploaded_file)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(input_image, caption="Input Image", use_container_width =True)

    st.subheader("Enter Parameters (Single Float Values)")
    a_temp = st.number_input("X start", value=0.0)
    b_temp = st.number_input("X End", value=10.0)
    i_temp = st.number_input("Y start", value=0.0)
    j_temp = st.number_input("Y End", value=10.0)

    if st.button("Process"):
        output_image, excel_buffer = main(input_image, a_temp, b_temp, i_temp, j_temp)

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Input Image", use_container_width=True)
        with col2:
            st.image(output_image, caption="Output Image", use_container_width=True)

        st.download_button(
            label="Download Excel",
            data=excel_buffer,
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
