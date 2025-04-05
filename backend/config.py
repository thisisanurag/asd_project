# config.py

class Config:
    DEBUG = False
    TESTING = False
    HOST = "0.0.0.0"
    PORT = 5000
    YOLO_MODEL_PATH = "./best_yolov8_augmented.pt"
    UPLOAD_FOLDER = "uploads"
    ANNOTATED_FOLDER = "annotated"
    ALLOWED_ORIGINS = ["http://localhost:3000"]

class DevelopmentConfig(Config):
    DEBUG = True
    YOLO_MODEL_PATH = "./best_yolov8_augmented_dev.pt"  # Example path for development

class ProductionConfig(Config):
    DEBUG = False
    HOST = "0.0.0.0"
    PORT = 5000
    YOLO_MODEL_PATH = "./best_yolov8_augmented_prod.pt"  # Example path for production
    ALLOWED_ORIGINS = ["https://your-production-frontend-domain.com"]
