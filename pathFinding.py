import numpy as np
import heapq
import cv2
import matplotlib.pyplot as plt
from collections import deque
import time
import tensorflow as tf

class BuildingMap:
    def __init__(self, floor_plan=None, resolution=0.1):
        self.resolution = resolution
        self.floor_plan = None
        self.occupancy_grid = None
        self.risk_map = None
        self.height = 0
        self.width = 0
        
        if floor_plan is not None:
            self.load_floor_plan(floor_plan)
    
    def load_floor_plan(self, floor_plan):
        self.floor_plan = floor_plan
        self.height, self.width = floor_plan.shape[:2]
        
        self.occupancy_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        self.occupancy_grid[floor_plan < 128] = 1
        self.risk_map = np.zeros((self.height, self.width), dtype=np.float32)
    
    def update_from_slam(self, slam_data):
        for point in slam_data['obstacles']:
            x, y = self.world_to_grid(point)
            if 0 <= x < self.width and 0 <= y < self.height:
                self.occupancy_grid[y, x] = 1
    
    def update_risk(self, hazards):
        self.risk_map *= 0.95
        
        for hazard in hazards:
            h_type = hazard['type']
            x, y = self.world_to_grid(hazard['position'])
            intensity = hazard['intensity']
            radius = int(hazard['radius'] / self.resolution)
            
            if h_type == 'fire':
                weight = 1.0
            elif h_type == 'smoke':
                weight = 0.7
            else:
                weight = 0.5
            
            y_indices, x_indices = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x_indices**2 + y_indices**2 <= radius**2
            
            x_min = max(0, x - radius)
            y_min = max(0, y - radius)
            x_max = min(self.width, x + radius + 1)
            y_max = min(self.height, y + radius + 1)
            
            mask_x_min = radius - (x - x_min)
            mask_y_min = radius - (y - y_min)
            mask_x_max = mask_x_min + (x_max - x_min)
            mask_y_max = mask_y_min + (y_max - y_min)
            
            risk_values = np.exp(-((np.arange(x_min, x_max) - x)**2 + 
                                  (np.arange(y_min, y_max).reshape(-1, 1) - y)**2) / 
                                  (2 * (radius/2)**2)) * intensity * weight
            
            self.risk_map[y_min:y_max, x_min:x_max] = np.maximum(
                self.risk_map[y_min:y_max, x_min:x_max],
                risk_values
            )
    
    def world_to_grid(self, point):
        x, y = point
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, point):
        x, y = point
        world_x = x * self.resolution
        world_y = y * self.resolution
        return world_x, world_y
    
    def get_combined_cost_map(self, risk_weight=0.7):
        cost_map = np.ones_like(self.occupancy_grid, dtype=np.float32)
        cost_map[self.occupancy_grid == 1] = float('inf')
        cost_map += self.risk_map * risk_weight
        
        return cost_map


class PathFinder:
    def __init__(self, building_map):
        self.building_map = building_map
        self.directions = [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1)
        ]
        
        self.costs = [1, 1, 1, 1, 1.414, 1.414, 1.414, 1.414]
    
    def find_path(self, start, goal, risk_weight=0.7, max_iterations=10000):
        cost_map = self.building_map.get_combined_cost_map(risk_weight)
        
        if (cost_map[start[1], start[0]] == float('inf') or 
            cost_map[goal[1], goal[0]] == float('inf')):
            return None
        
        open_set = []
        closed_set = set()
        
        heapq.heappush(open_set, (self._heuristic(start, goal), 0, start, 0, None))
        
        counter = 1
        
        parents = {}
        g_costs = {start: 0}
        
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            _, _, current, current_g, parent = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if parent:
                parents[current] = parent
            
            if current == goal:
                return self._reconstruct_path(parents, goal)
            
            closed_set.add(current)
            
            for i, (dx, dy) in enumerate(self.directions):
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                if (nx < 0 or nx >= self.building_map.width or 
                    ny < 0 or ny >= self.building_map.height):
                    continue
                
                if neighbor in closed_set:
                    continue
                
                if cost_map[ny, nx] == float('inf'):
                    continue
                
                if i >= 4:
                    if (cost_map[current[1], nx] == float('inf') or 
                        cost_map[ny, current[0]] == float('inf')):
                        continue
                
                move_cost = self.costs[i] * cost_map[ny, nx]
                new_g = current_g + move_cost
                
                if neighbor not in g_costs or new_g < g_costs[neighbor]:
                    g_costs[neighbor] = new_g
                    f_cost = new_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_cost, counter, neighbor, new_g, current))
                    counter += 1
        
        return None
    
    def _heuristic(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    
    def _reconstruct_path(self, parents, current):
        path = [current]
        while current in parents:
            current = parents[current]
            path.append(current)
        
        return list(reversed(path))
    
    def smooth_path(self, path, smoothing_factor=0.5):
        if path is None or len(path) <= 2:
            return path
        
        smoothed_path = [path[0]]
        
        for i in range(1, len(path) - 1):
            prev = smoothed_path[-1]
            current = path[i]
            next_point = path[i + 1]
            
            smoothed_x = current[0] + smoothing_factor * (prev[0] + next_point[0] - 2 * current[0])
            smoothed_y = current[1] + smoothing_factor * (prev[1] + next_point[1] - 2 * current[1])
            
            smoothed_path.append((int(smoothed_x), int(smoothed_y)))
        
        smoothed_path.append(path[-1])
        return smoothed_path


class DynamicPathPlanner:
    def __init__(self, building_map):
        self.building_map = building_map
        self.path_finder = PathFinder(building_map)
        self.current_path = None
        self.goal = None
        self.replan_threshold = 0.3
        self.last_plan_time = 0
        self.min_replan_interval = 2.0
    
    def set_goal(self, goal):
        self.goal = goal
    
    def update(self, current_position, hazards=None):
        current_time = time.time()
        
        if hazards:
            self.building_map.update_risk(hazards)
        
        if self.goal is None:
            return None
        
        cost_map = self.building_map.get_combined_cost_map()
        
        replan_needed = False
        
        if self.current_path is None:
            replan_needed = True
        elif current_time - self.last_plan_time > self.min_replan_interval:
            if len(self.current_path) > 1:
                next_point = self.current_path[1]
                if cost_map[next_point[1], next_point[0]] > self.replan_threshold:
                    replan_needed = True
        
        if replan_needed:
            self.current_path = self.path_finder.find_path(current_position, self.goal)
            self.current_path = self.path_finder.smooth_path(self.current_path)
            self.last_plan_time = current_time
        
        return self.current_path


class RescueNavigation:
    def __init__(self):
        self.building_map = None
        self.path_planner = None
        self.victim_detector = None
        self.exit_detector = None
        self.hazard_detector = None
    
    def initialize_from_floor_plan(self, floor_plan):
        self.building_map = BuildingMap(floor_plan)
        self.path_planner = DynamicPathPlanner(self.building_map)
        self.initialize_detectors()
    
    def initialize_from_scratch(self):
        self.building_map = BuildingMap()
        self.path_planner = DynamicPathPlanner(self.building_map)
        self.initialize_detectors()
    
    def initialize_detectors(self):
        self.victim_detector = self._load_victim_detector()
        self.exit_detector = self._load_exit_detector()
        self.hazard_detector = self._load_hazard_detector()
    
    def _load_victim_detector(self):
        model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        return model
    
    def _load_exit_detector(self):
        return None
    
    def _load_hazard_detector(self):
        return None
    
    def update_from_sensors(self, sensor_data):
        if 'slam' in sensor_data:
            self.building_map.update_from_slam(sensor_data['slam'])
        
        hazards = []
        
        if 'thermal' in sensor_data:
            fire_hazards = self._detect_fire(sensor_data['thermal'])
            hazards.extend(fire_hazards)
        
        if 'visual' in sensor_data:
            smoke_hazards = self._detect_smoke(sensor_data['visual'])
            hazards.extend(smoke_hazards)
        
        if hazards:
            self.building_map.update_risk(hazards)
    
    def _detect_fire(self, thermal_data):
        hazards = []
        
        return hazards
    
    def _detect_smoke(self, visual_data):
        hazards = []
        
        return hazards
    
    def detect_victims(self, visual_data, thermal_data):
        victims = []
        
        return victims
    
    def plan_path_to_victim(self, current_position, victim_position):
        self.path_planner.set_goal(victim_position)
        return self.path_planner.update(current_position)
    
    def plan_path_to_exit(self, current_position, exit_position):
        self.path_planner.set_goal(exit_position)
        return self.path_planner.update(current_position)
    
    def get_navigation_instructions(self, path, drone_position, drone_heading):
        if path is None or len(path) < 2:
            return None
        
        instructions = []
        current_idx = 0
        
        for i, point in enumerate(path):
            if i == 0:
                continue
            
            prev_point = path[i-1]
            
            dx = point[0] - prev_point[0]
            dy = point[1] - prev_point[1]
            
            distance = ((dx ** 2 + dy ** 2) ** 0.5) * self.building_map.resolution
            
            angle = np.arctan2(dy, dx) * 180 / np.pi
            rel_angle = angle - drone_heading
            
            if rel_angle > 180:
                rel_angle -= 360
            elif rel_angle < -180:
                rel_angle += 360
            
            instructions.append({
                'point': self.building_map.grid_to_world(point),
                'angle': angle,
                'rel_angle': rel_angle,
                'distance': distance
            })
        
        return instructions
    
    def visualize_path(self, path):
        if path is None:
            return None
        
        vis_map = np.zeros((self.building_map.height, self.building_map.width, 3), dtype=np.uint8)
        
        # Background: Occupancy grid
        vis_map[self.building_map.occupancy_grid == 0] = [50, 50, 50]
        vis_map[self.building_map.occupancy_grid == 1] = [0, 0, 0]
        
        # Risk overlay
        risk_overlay = cv2.applyColorMap(
            (self.building_map.risk_map * 255).astype(np.uint8),
            cv2.COLORMAP_HOT
        )
        mask = self.building_map.risk_map > 0.1
        vis_map[mask] = risk_overlay[mask]
        
        # Draw path
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i+1]
            cv2.line(vis_map, p1, p2, (0, 255, 0), 2)
        
        # Mark start and goal
        cv2.circle(vis_map, path[0], 5, (0, 0, 255), -1)
        cv2.circle(vis_map, path[-1], 5, (255, 0, 0), -1)
        
        return vis_map


def demo():
    # 샘플 바이너리 이미지 생성 (흰색 = 자유 공간, 검은색 = 장애물)
    floor_plan = np.ones((100, 100), dtype=np.uint8) * 255
    
    # 벽 생성
    floor_plan[20:80, 20] = 0
    floor_plan[20:80, 80] = 0
    floor_plan[20, 20:80] = 0
    floor_plan[80, 20:80] = 0
    
    # 내부 장애물
    floor_plan[40:60, 20:40] = 0
    floor_plan[30:50, 60:70] = 0
    
    # BuildingMap 초기화
    building_map = BuildingMap(floor_plan)
    
    # 화재 위험 추가
    hazards = [
        {'type': 'fire', 'position': (30, 30), 'intensity': 1.0, 'radius': 10},
        {'type': 'smoke', 'position': (60, 70), 'intensity': 0.7, 'radius': 15}
    ]
    building_map.update_risk(hazards)
    
    # PathFinder 초기화
    path_finder = PathFinder(building_map)
    
    # 시작점과 목표점 설정
    start = (25, 25)
    goal = (75, 75)
    
    # 경로 탐색
    path = path_finder.find_path(start, goal)
    
    # 결과 시각화
    vis_map = np.zeros((building_map.height, building_map.width, 3), dtype=np.uint8)
    
    # 바닥 맵
    vis_map[floor_plan == 255] = [255, 255, 255]
    vis_map[floor_plan == 0] = [0, 0, 0]
    
    # 위험 맵
    max_risk = np.max(building_map.risk_map)
    if max_risk > 0:
        risk_norm = building_map.risk_map / max_risk
        risk_vis = (risk_norm * 255).astype(np.uint8)
        risk_color = cv2.applyColorMap(risk_vis, cv2.COLORMAP_HOT)
        for y in range(building_map.height):
            for x in range(building_map.width):
                if building_map.risk_map[y, x] > 0.1:
                    vis_map[y, x] = risk_color[y, x]
    
    # 경로 그리기
    if path:
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            cv2.line(vis_map, p1, p2, (0, 255, 0), 2)
        
        # 시작점과 목표점 표시
        cv2.circle(vis_map, start, 5, (0, 0, 255), -1)
        cv2.circle(vis_map, goal, 5, (255, 0, 0), -1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_map)
    plt.title("ResQon 경로 탐색 결과")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    demo()