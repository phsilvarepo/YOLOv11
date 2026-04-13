import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import os
from ultralytics import YOLO

class Yolov11Node(Node):
    def __init__(self):
        super().__init__('generic_yolo_node')
        self.bridge = CvBridge()

        # 1. Config from Env (Dashboard injected)
        model_path = os.environ.get('MODEL_PATH', 'yolo11n.pt')
        self.conf = float(os.environ.get('CONFIDENCE_THRESHOLD', '0.5'))
        self.imgsz = int(os.environ.get('IMAGE_RESOLUTION', '640'))
        input_topic = os.environ.get('INPUT_TOPIC', '/image_raw')

        # 2. Initialize Model
        self.get_logger().info(f"Loading Model: {model_path}")
        self.model = YOLO(model_path)
        
        # 3. Dynamic Publishers based on Dashboard Envs
        self.img_pub = self.create_publisher(Image, os.environ['OUTPUT_TOPIC_IMAGE'], 10) if 'OUTPUT_TOPIC_IMAGE' in os.environ else None
        self.bb_pub = self.create_publisher(Detection2DArray, os.environ['OUTPUT_TOPIC_BB'], 10) if 'OUTPUT_TOPIC_BB' in os.environ else None
        self.marker_pub = self.create_publisher(MarkerArray, os.environ['OUTPUT_TOPIC_MARKERS'], 10) if 'OUTPUT_TOPIC_MARKERS' in os.environ else None

        # 4. Subscriber
        self.subscription = self.create_subscription(Image, input_topic, self.image_callback, 10)
        self.get_logger().info(f"YOLO Generic Node active. Task: {self.model.task}")

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run inference
        results = self.model(cv_image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]

        # --- Output 1: Debug Image (Supports Detect, Seg, and Pose automatically) ---
        if self.img_pub:
            # results.plot() draws boxes, masks, or skeletons based on the .pt file used
            annotated_frame = results.plot()
            img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            img_msg.header = msg.header
            self.img_pub.publish(img_msg)

        # --- Output 2: Bounding Boxes (FIXED) ---
        if self.bb_pub:
            bb_msg = Detection2DArray()
            bb_msg.header = msg.header
            
            # Only loop if boxes actually exist, but ALWAYS publish the msg
            if results.boxes:
                for box in results.boxes:
                    det = Detection2D()
                    det.bbox.center.position.x = float(box.xywh[0][0])
                    det.bbox.center.position.y = float(box.xywh[0][1])
                    det.bbox.size_x = float(box.xywh[0][2])
                    det.bbox.size_y = float(box.xywh[0][3])

                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = self.model.names[int(box.cls[0])]
                    hyp.hypothesis.score = float(box.conf[0])
                    det.results.append(hyp)
                    bb_msg.detections.append(det)
            
            # This now sends an empty detections list [] when results.boxes is empty
            self.bb_pub.publish(bb_msg)

        # --- Output 3: Pose Markers (Specific to Pose models) ---
        if self.marker_pub and results.keypoints:
            marker_array = MarkerArray()
            for i, person_kpts in enumerate(results.keypoints.xy):
                marker = Marker()
                marker.header = msg.header
                marker.ns = "yolo_pose"
                marker.id = i
                marker.type = Marker.SPHERE_LIST
                marker.action = Marker.ADD
                marker.scale.x = marker.scale.y = marker.scale.z = 0.05
                marker.color.r, marker.color.a = 1.0, 1.0
                
                for kp in person_kpts:
                    if kp[0] == 0 and kp[1] == 0: continue # Skip untracked points
                    p = Point()
                    p.x, p.y, p.z = float(kp[0]), float(kp[1]), 0.0
                    marker.points.append(p)
                marker_array.markers.append(marker)
            self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = Yolov11Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
