
import carla
import queue
import time
from ultralytics import YOLO
import numpy as np
import cv2
import os
import random
import traceback

# === Configuration ===

# List of YOLOv8 model paths to use for detection
MODEL_PATHS = [
    r"C:\thowl\AUV\Carla\yolov8n.pt",
    r"C:\thowl\AUV\Carla\REAL.pt"
]

# Where to save the annotated video output
SAVE_IMAGE_DIR = r"C:\thowl\AUV\Carla\bestmodel"
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)

# Mapping class IDs to readable names (0 = person, 2 = car)
CLASS_NAMES = {0: "person", 1: "car"}


# === Utility Functions ===

def preprocess_image(image):
    """
    Converts raw CARLA image data to an RGB image suitable for YOLO.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # RGBA format
    image_bgr = array[:, :, :3]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def non_max_suppression_fusion(detections, iou_threshold=0.5):
    """
    Applies non-maximum suppression (NMS) to remove duplicate detections from multiple models.
    Keeps only people and cars (class IDs 0 and 2).
    """
    if len(detections) == 0:
        return []

    boxes = []
    confidences = []
    class_ids = []

    # Filter and reformat detections
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) not in [0, 2]:  # Only keep 'person' and 'car'
            continue
        boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # Format for OpenCV
        confidences.append(float(conf))
        class_ids.append(int(cls))

    # Run OpenCV's NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, iou_threshold)

    fused = []
    if len(indices) > 0:
        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[i]
            fused.append([x, y, x + w, y + h, confidences[i], class_ids[i]])

    return fused


def spawn_traffic(world, blueprint_library, traffic_manager, spawn_points, num_vehicles=20):
    """
    Spawns a set of autonomous traffic vehicles at random spawn points.
    """
    ALLOWED_VEHICLES = [
        "vehicle.audi.tt",
        "vehicle.tesla.model3",
        "vehicle.bmw.grandtourer",
        "vehicle.mercedes.coupe",
        "vehicle.chevrolet.impala"
    ]

    # Filter only desired vehicle blueprints
    vehicle_bps = [bp for bp in blueprint_library.filter("vehicle.*") if bp.id in ALLOWED_VEHICLES]

    vehicles = []
    for spawn_point in random.sample(spawn_points, min(num_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            vehicles.append(vehicle)

    return vehicles


def spawn_pedestrians(world, blueprint_library, number_of_pedestrians):
    """
    Spawns a number of AI pedestrians with random walking behavior.
    """
    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    controller_bp = blueprint_library.find('controller.ai.walker')

    walkers = []
    controllers = []

    for i in range(number_of_pedestrians):
        spawn_location = world.get_random_location_from_navigation()
        if spawn_location is None:
            print(f"[{i}] No valid spawn location. Skipping.")
            continue

        spawn_location.z += 1  # Raise slightly above the ground
        transform = carla.Transform(spawn_location)

        walker_bp = random.choice(walker_bps)
        walker = world.try_spawn_actor(walker_bp, transform)
        if walker is None:
            print(f"[{i}] Failed to spawn walker.")
            continue

        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is None:
            print(f"[{i}] Failed to spawn controller. Destroying walker.")
            walker.destroy()
            continue

        walkers.append(walker)
        controllers.append(controller)
        print(f"Total walkers: {len(walkers)}, controllers: {len(controllers)}")
        world.tick()

    # Start every other controller and assign random destinations
    for i in range(0, len(walkers), 2):
        walker = walkers[i]
        controller = controllers[i]
        controller.start()
        destination = world.get_random_location_from_navigation()
        if destination:
            controller.go_to_location(destination)

    return walkers, controllers


def destroy_all_actors(actors, world, settings):
    """
    Cleans up all spawned actors and resets the world settings.
    """
    for actor in actors:
        try:
            if actor is not None:
                actor.destroy()
        except Exception as e:
            print(f"Failed to destroy actor {actor.id if actor else 'unknown'}: {e}")

    print("All applicable actors destroyed.")

    # Reset simulation mode to avoid locking CARLA
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)


# === Main Logic ===

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    # Load town and get blueprint library
    world = client.load_world("Town02")
    blueprint_library = world.get_blueprint_library()

    # Enable synchronous mode for better control
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # ~20 FPS
    world.apply_settings(settings)

    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)

    spawn_points = world.get_map().get_spawn_points()

    # Spawn ego vehicle
    ego_bp = blueprint_library.find("vehicle.tesla.model3")
    ego_spawn_point = random.choice(spawn_points)
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    if not ego_vehicle:
        raise RuntimeError("Failed to spawn ego vehicle")
    ego_vehicle.set_autopilot(True, tm.get_port())

    # Add traffic and pedestrians
    traffic_vehicles = spawn_traffic(world, blueprint_library, tm, spawn_points)
    pedestrians, controllers = spawn_pedestrians(world, blueprint_library, number_of_pedestrians=50)

    # Attach RGB camera to ego vehicle
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "1920")
    camera_bp.set_attribute("image_size_y", "1080")
    camera_bp.set_attribute("fov", "90")
    camera_transform = carla.Transform(carla.Location(x=1, z=2.5))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

    # Set up image queue for camera frames
    image_queue = queue.Queue()
    camera.listen(lambda image: image_queue.put(image))

    # Load YOLO models
    models = [YOLO(path) for path in MODEL_PATHS]

    frame_count = 0
    video_save_path = os.path.join(SAVE_IMAGE_DIR, "REALnew.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    frame_size = (1920, 1080)
    video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, frame_size)

    try:
        while True:
            world.tick()

            try:
                image = image_queue.get(timeout=1.0)
            except queue.Empty:
                print("Warning: No image this frame.")
                continue

            rgb_image = preprocess_image(image)

            try:
                all_detections = []
                for model in models:
                    results = model.predict(source=rgb_image, imgsz=512, save=False, conf=0.25)
                    boxes = results[0].boxes
                    if boxes is not None and boxes.data is not None:
                        all_detections.extend(boxes.data.cpu().numpy())

                # Fuse detections from both models
                fused = non_max_suppression_fusion(all_detections)

                # Draw annotations on a copy of the image
                annotated_image = rgb_image.copy()
                for det in fused:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) in CLASS_NAMES:
                        label = CLASS_NAMES[int(cls)]
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(annotated_image, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Convert to BGR for OpenCV video writing
                bgr_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
                print(f"[Frame {frame_count}] Written")

            except Exception:
                print("⚠️ Detection error:")
                traceback.print_exc()

            frame_count += 1

    finally:
        # Clean up resources
        print("Cleaning up...")
        camera.stop()
        video_writer.release()
        destroy_all_actors([ego_vehicle] + traffic_vehicles + pedestrians + controllers, world, settings)
        print("✅ Done. Video saved to:", video_save_path)


if __name__ == "__main__":
    main()
