import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from fpdf import FPDF
import io

def detect_lines(image_bytes, angle_tolerance=10, eps=5, min_samples=2):
    try:
        
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
        
        
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        
        adaptive_thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        
        edges = cv2.Canny(adaptive_thresh, 50, 100, apertureSize=3)

        
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        horizontal_lines = []
        vertical_lines = []

        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                degree_angle = np.degrees(theta)
                if (degree_angle < 0.8 or degree_angle > 180 - 0.8):
                    vertical_lines.append(line)
                elif (90 - angle_tolerance < degree_angle < 90 + angle_tolerance):
                    horizontal_lines.append(line)

        def cluster_lines(lines, eps, min_samples):
            if not lines:
                return []
            points = np.array([[line[0][0] * np.cos(line[0][1]), line[0][0] * np.sin(line[0][1])] for line in lines])
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            unique_labels = set(clustering.labels_)
            return unique_labels, clustering

        vertical_labels, _ = cluster_lines(vertical_lines, eps, min_samples)
        horizontal_labels, _ = cluster_lines(horizontal_lines, eps, min_samples)
        horicount = len(horizontal_labels) - (1 if -1 in horizontal_labels else 0)
        verticount = len(vertical_labels) - (1 if -1 in vertical_labels else 0)
        total_count = verticount + horicount

        
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        _, img_encoded = cv2.imencode('.png', color_image)
        return img_encoded.tobytes(), verticount, horicount, total_count

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, 0, 0, 0

def create_pdf_report(tester_name, test_date, vertical_count, horizontal_count, total_count):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt="Thread Count Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Tester: {tester_name}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {test_date}", ln=True)
    pdf.cell(200, 10, txt=f"Warps: {vertical_count}", ln=True)
    pdf.cell(200, 10, txt=f"Wefts: {horizontal_count}", ln=True)
    pdf.cell(200, 10, txt=f"Total Threads: {total_count}", ln=True)

    
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    return pdf_output

def main():
    st.title("FabricSense App")

    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        
        result_image, vertical_count, horizontal_count, total_count = detect_lines(uploaded_file.read())

        if result_image:
            st.image(result_image, caption="Detected Lines", use_column_width=True)

            
            st.subheader("Enter Tester Information")
            tester_name = st.text_input("Enter tester name")
            test_date = st.text_input("Enter date")

            if st.button("Submit Info"):
                st.session_state.tester_name = tester_name
                st.session_state.test_date = test_date
                st.session_state.vertical_count = vertical_count
                st.session_state.horizontal_count = horizontal_count
                st.session_state.total_count = total_count
                st.success("Information submitted successfully!")

    if 'tester_name' in st.session_state and 'test_date' in st.session_state:
        st.subheader("Report")
        st.write(f"Tester: {st.session_state.tester_name}")
        st.write(f"Date: {st.session_state.test_date}")
        st.write(f"Vertical Threads: {st.session_state.vertical_count}")
        st.write(f"Horizontal Threads: {st.session_state.horizontal_count}")
        st.write(f"Total Threads: {st.session_state.total_count}")

        
        pdf = create_pdf_report(
            st.session_state.tester_name, 
            st.session_state.test_date, 
            st.session_state.vertical_count, 
            st.session_state.horizontal_count, 
            st.session_state.total_count
        )
        st.download_button(label="Download Report as PDF", data=pdf, file_name="report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()
