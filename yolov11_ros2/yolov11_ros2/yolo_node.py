import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from ultralytics import YOLO

class YoloV11Detector(Node):
    def __init__(self):
        super().__init__('yolov11_detector')
        
        # 1. Fetch values from Environment Variables (set by our Dashboard Backend)
        env_model = os.environ.get('MODEL_PATH', 'yolo11n.pt')
        env_input = os.environ.get('INPUT_TOPIC', '/rgb')
        env_output = os.environ.get('OUTPUT_TOPIC', '/yolo/detections_image')
        
        env_conf = os.environ.get('CONFIDENCE_THRESHOLD', '0.5')
        env_size = os.environ.get('IMAGE_RESOLUTION', '640')
        
        print(env_conf)

        # 2. Declare ROS parameters
        self.declare_parameter('model_path', env_model)
        self.declare_parameter('image_topic', env_input)
        self.declare_parameter('output_topic', env_output)
        self.declare_parameter('conf_threshold', float(env_conf))
        self.declare_parameter('img_size', int(env_size))
        
        # 3. Get the final values
        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.conf = self.get_parameter('conf_threshold').value
        self.imgsz = self.get_parameter('img_size').value

        self.get_logger().info(f"--- YOLOv11 Initialized ---")
        self.get_logger().info(f"Model: {self.model_path} | Res: {self.imgsz} | Conf: {self.conf}")
        self.get_logger().info(f"Sub: {self.image_topic} -> Pub: {self.output_topic}")

        # Load Model
        self.model = YOLO(self.model_path)
        self.bridge = CvBridge()

        # 4. ROS Subscriber and Publisher
        self.subscription = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        
        self.publisher = self.create_publisher(
            Image, self.output_topic, 10)
            
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run Inference with our Dynamic Params
        results = self.model(
            cv_image, 
            conf=self.conf, 
            imgsz=self.imgsz, 
            verbose=False
        )
        
        # Plot and Publish
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
