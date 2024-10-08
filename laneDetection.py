import cv2
import numpy as np

def edge_detection(image):
    if image is None:
        video.release()
        cv2.destroyAllWindows()
        exit()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k_size = 5
    blurred = cv2.GaussianBlur(grayscale, (k_size, k_size), 0)
    edges = cv2.Canny(grayscale, 50, 150)
    return edges

def roi(edges):
    img_height = edges.shape[0]
    img_width = edges.shape[1]
    blank_mask = np.zeros_like(edges)
    polygon = np.array([[
        (200, img_height),
        (800, 350),
        (1200, img_height),]], np.int32)
    cv2.fillPoly(blank_mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, blank_mask)
    return cropped_edges

def detect_lines(roi_edges):
    return cv2.HoughLinesP(roi_edges, 2, np.pi / 180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)

def blend_images(original, line_overlay):
    return cv2.addWeighted(original, 0.8, line_overlay, 1, 1)

def draw_lines(image, lines):
    lines_img = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return lines_img

def calculate_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y_bottom = int(image.shape[0])
    y_top = int(y_bottom * 3.0 / 5)
    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return [[x_bottom, y_bottom, x_top, y_top]]

def compute_avg_slope(image, detected_lines):
    left_lines = []
    right_lines = []
    if detected_lines is None:
        return None
    for line in detected_lines:
        for x1, y1, x2, y2 in line:
            fit_params = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit_params[0]
            intercept = fit_params[1]
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))
    left_avg = np.average(left_lines, axis=0)
    right_avg = np.average(right_lines, axis=0)
    left_final_line = calculate_coordinates(image, left_avg)
    right_final_line = calculate_coordinates(image, right_avg)
    return [left_final_line, right_final_line]

video = cv2.VideoCapture("test1.mp4")
while(video.isOpened()):
    _, frame = video.read()
    edge_img = edge_detection(frame)
    cropped_edge = roi(edge_img)

    detected_lines = detect_lines(cropped_edge)
    avg_lines = compute_avg_slope(frame, detected_lines)
    line_overlay = draw_lines(frame, avg_lines)
    output = blend_images(frame, line_overlay)
    cv2.imshow("Lane Detection", output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
