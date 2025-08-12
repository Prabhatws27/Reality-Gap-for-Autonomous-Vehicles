import carla 
import random
import time
import numpy as np
import cv2
import queue
import os

# Save dirs
SAVE_IMAGE_DIR = r"C:\thowl\AUV\Carla\dataset\output8\images"
SAVE_LABEL_DIR = r"C:\thowl\AUV\Carla\dataset\output8\labels"

os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
os.makedirs(SAVE_LABEL_DIR, exist_ok=True)

# Semantic class IDs for detection
SEMANTIC_CLASSES = {
    12: 0,  # pedestrian
    14: 1,  # car
}

# Converts BGRA image to BGR image (usable for OpenCV and YOLO model)
def carla_image_to_numpy(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)  
    array = array.reshape((image.height, image.width, 4))
    return cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGRA2BGR)

# Returns Blue channel from the BGRA seg image
def carla_seg_to_semantic_channel(seg_image):
    array = np.frombuffer(seg_image.raw_data, dtype=np.uint8)
    array = array.reshape((seg_image.height, seg_image.width, 4))
    return array[:, :, 2] 

#Spawning Pedestrians
def spawn_pedestrians(world, blueprint_library, number_of_pedestrians):
    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    controller_bp = blueprint_library.find('controller.ai.walker')

    walkers = []
    controllers = []

    for i in range(number_of_pedestrians):
        spawn_location = world.get_random_location_from_navigation()

        if spawn_location is None:
            print(f"[{i}] No valid spawn location. Skipping.")
            continue

        spawn_location.z += 1
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

    for i in range(0, len(walkers), 2):
        walker = walkers[i]
        controller = controllers[i]
        controller.start()
        destination = world.get_random_location_from_navigation()
        if destination:
            controller.go_to_location(destination)

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.load_world("Town10HD")

    world.set_weather(carla.WeatherParameters.ClearNoon)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprints = blueprint_library.filter("vehicle.*")
    bike_blueprints = blueprint_library.filter("vehicle.bike*")

    # Traffic Manager
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm_port = tm.get_port()

    # Ego vehicle
    ego_bp = blueprint_library.find("vehicle.tesla.model3")
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn = random.choice(spawn_points)
    ego_vehicle = world.spawn_actor(ego_bp, ego_spawn)
    ego_vehicle.set_autopilot(True, tm_port)

    # Vehicles and bikes
    spawned_vehicles = []
    vehicle_spawn_points = random.sample(spawn_points, 50)
    for spawn_point in vehicle_spawn_points:
        if random.random() < 0.6 and bike_blueprints:
            bp = random.choice(bike_blueprints)
        else:
            bp = random.choice(vehicle_blueprints)
        vehicle = world.try_spawn_actor(bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True, tm_port)
            spawned_vehicles.append(vehicle)

    # Replace old walker spawning with improved pedestrian spawning
    spawn_pedestrians(world, blueprint_library, 100)

    # RGB cam
    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1920")
    cam_bp.set_attribute("image_size_y", "1080")
    cam_bp.set_attribute("fov", "90")
    cam_transform = carla.Transform(carla.Location(x=1, z=2.5)) 
    ego_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)

    # Semantic Cam
    seg_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
    seg_bp.set_attribute("image_size_x", "1920")
    seg_bp.set_attribute("image_size_y", "1080")
    seg_bp.set_attribute("fov", "90")
    ego_seg_cam = world.spawn_actor(seg_bp, cam_transform, attach_to=ego_vehicle)

    rgb_queue = queue.Queue()
    seg_queue = queue.Queue()
    ego_cam.listen(lambda image: rgb_queue.put(image))
    ego_seg_cam.listen(lambda image: seg_queue.put(image))

    frame = 0
    running = True
    last_save_time = time.time()
    
    
    try:
        while running:
            world.tick()

            try:
                rgb_image = rgb_queue.get(timeout=1.0)
                seg_image = seg_queue.get(timeout=1.0)
            except queue.Empty:
                print("Warning: Sensor data timeout.")
                continue

            rgb_array = carla_image_to_numpy(rgb_image)
            semantic_channel = carla_seg_to_semantic_channel(seg_image)

            h, w, _ = rgb_array.shape

            bboxes_by_class = {cid: [] for cid in SEMANTIC_CLASSES.values()}
            for carla_id, yolo_id in SEMANTIC_CLASSES.items():
                mask = (semantic_channel == carla_id).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if bw * bh > 150:
                        bboxes_by_class[yolo_id].append((x, y, x + bw, y + bh))
                        color = (0, 255, 0) if yolo_id == 1 else (255, 0, 0) if yolo_id == 2 else (0, 0, 255)
                        cv2.rectangle(rgb_array, (x, y), (x + bw, y + bh), color, 2)

            current_time = time.time()
            if current_time - last_save_time >= 1.0:
                img_path = os.path.join(SAVE_IMAGE_DIR, f"frame_{frame:04d}.png")
                label_path = os.path.join(SAVE_LABEL_DIR, f"frame_{frame:04d}.txt")
                cv2.imwrite(img_path, rgb_array)
                with open(label_path, "w") as f:
                    for class_id, boxes in bboxes_by_class.items():
                        for (x1, y1, x2, y2) in boxes:
                            cx = ((x1 + x2) / 2) / w
                            cy = ((y1 + y2) / 2) / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                print(f"Saved frame {frame}")
                last_save_time = current_time
                frame += 1

    finally:
        print("Cleaning up...")
        ego_cam.stop()
        ego_seg_cam.stop()
        tm.set_synchronous_mode(False)
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))

        actors = spawned_vehicles + [ego_cam, ego_seg_cam, ego_vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        print("All cleaned up.")

if __name__ == "__main__":
    main()
