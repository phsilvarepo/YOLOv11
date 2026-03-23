import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os  # <--- ADD THIS IMPORT
from ultralytics import YOLO

class YoloV11Detector(Node):
    def __init__(self):
        super().__init__('yolov11_detector')
        
        # 1. Fetch values from Environment Variables (Docker) 
        # or use defaults if the ENV vars don't exist
        env_model = os.environ.get('YOLO_MODEL', 'yolo11n.pt')
        env_input = os.environ.get('YOLO_INPUT_TOPIC', '/rgb')
        env_output = os.environ.get('YOLO_OUTPUT_TOPIC', '/yolo/detections_image')

        # 2. Declare ROS parameters using those values
        self.declare_parameter('model_path', env_model)
        self.declare_parameter('image_topic', env_input)
        self.declare_parameter('output_topic', env_output)
        
        # 3. Get the final values
        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.get_logger().info(f"Loading YOLOv11 model: {model_path}")
        self.get_logger().info(f"Subscribed to: {image_topic}")
        self.get_logger().info(f"Publishing to: {output_topic}")

        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # 4. Use the variables in the Subscriber and Publisher
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
        
        self.publisher = self.create_publisher(
            Image, output_topic, 10)
            
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        out_msg.header = msg.header 
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloV11Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()