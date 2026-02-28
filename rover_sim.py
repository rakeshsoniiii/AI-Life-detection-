from ursina import *
import random
import time

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
WINDOW_TITLE = "AI DISASTER RESCUE ROVER - ADVANCED 3D SIMULATION"
ROVER_SPEED = 6
DETECTION_RANGE = 12
BATTERY_DRAIN_BASE = 0.01
BODY_NEAR_DISTANCE = 1.0

app = Ursina(title=WINDOW_TITLE, borderless=False)

# ==========================================
# ENTITIES (Survivor, Injured, Dead)
# ==========================================

class RescueTarget(Entity):
    def __init__(self, target_type='survivor', **kwargs):
        # survivor: green, injured: yellow, dead: red
        target_colors = {'survivor': color.green, 'injured': color.yellow, 'dead': color.red}
        # Root entity acts as a holder with a box collider so the raycast can hit it.
        super().__init__(model=None, collider='box', **kwargs)
        self.target_type = target_type
        self.detected = False
        self.confidence = 0.0
        base_color = target_colors.get(target_type, color.white)
        
        # Simple humanoid made from basic shapes
        self.body = Entity(parent=self, model='cube', color=base_color, scale=(0.6, 1.0, 0.3), y=0.9)
        self.head = Entity(parent=self, model='sphere', color=color.white, scale=0.35, y=1.55)
        self.left_arm = Entity(parent=self, model='cube', color=base_color, scale=(0.18, 0.8, 0.18), x=-0.45, y=0.95)
        self.right_arm = Entity(parent=self, model='cube', color=base_color, scale=(0.18, 0.8, 0.18), x=0.45, y=0.95)
        leg_color = color.rgb(60, 60, 60)
        self.left_leg = Entity(parent=self, model='cube', color=leg_color, scale=(0.2, 0.9, 0.25), x=-0.18, y=0.35)
        self.right_leg = Entity(parent=self, model='cube', color=leg_color, scale=(0.2, 0.9, 0.25), x=0.18, y=0.35)
        
        # Label above target
        self.label = Text(text='', parent=self, y=2.2, scale=10, origin=(0,0), color=base_color, enabled=False)

    def mark_detected(self):
        self.detected = True
        self.label.text = self.target_type.upper()
        self.label.enabled = True
        highlight = color.lime if self.target_type != 'dead' else color.red
        self.body.color = highlight
        self.left_arm.color = highlight
        self.right_arm.color = highlight

# ==========================================
# ASSETS & MODELS (PROCEDURAL TANK)
# ==========================================

class TankRover(Entity):
    def __init__(self, **kwargs):
        super().__init__(model='cube', color=color.gray, scale=(1.5, 0.8, 2), **kwargs)
        
        # Tank Body Details
        self.chassis = Entity(parent=self, model='cube', color=color.dark_gray, scale=(1.1, 0.5, 1.1), y=0.5)
        self.turret = Entity(parent=self.chassis, model='cube', color=color.black, scale=(0.6, 0.4, 0.6), y=0.5)
        self.barrel = Entity(parent=self.turret, model='cube', color=color.black, scale=(0.15, 0.15, 1.2), z=0.5, y=0.1)
        
        # Treads
        self.left_tread = Entity(parent=self, model='cube', color=color.black, scale=(0.3, 0.6, 1.05), x=-0.6, y=-0.1)
        self.right_tread = Entity(parent=self, model='cube', color=color.black, scale=(0.3, 0.6, 1.05), x=0.6, y=-0.1)
        
        # Mission Logic State
        self.battery = 100.0
        self.mode = 'MANUAL' # 'AUTO', 'SEMI-AUTO', 'MANUAL'
        self.status = "SYSTEM READY"
        self.comms_lost = False
        self.pause_until = 0.0  # time-based pause after detections
        self.detected_count = {'survivor': 0, 'injured': 0, 'dead': 0}
        self.collision_alert = False
        self.target_pos = None
        self.base_pos = self.position
        self.returning_to_base = False

    def update_logic(self, env):
        # If we're in a temporary pause (eg. just found a body), skip movement logic
        if time.time() < self.pause_until:
            return

        # Fail-Safe: Comms Loss
        if self.comms_lost:
            self.returning_to_base = True
            self.mode = 'AUTO'
            self.status = "SIGNAL LOST - RETURNING TO BASE"
            self.target_pos = self.base_pos
        
        # Fail-Safe: Battery
        if self.battery < 20:
            self.returning_to_base = True
            self.mode = 'AUTO'
            self.status = "LOW BATTERY - RETURNING TO BASE"
            self.target_pos = self.base_pos
        
        move_vec = Vec3(0,0,0)
        
        if self.mode in ['MANUAL', 'SEMI-AUTO'] and not self.returning_to_base:
            # Movement Input
            forward = held_keys['w'] or held_keys['up arrow']
            backward = held_keys['s'] or held_keys['down arrow']
            left = held_keys['a'] or held_keys['left arrow']
            right = held_keys['d'] or held_keys['right arrow']
            brake = held_keys['space']
            
            v_dir = (1 if forward else 0) - (1 if backward else 0)
            turn_dir = (1 if right else 0) - (1 if left else 0)

            rotation_speed = 90
            self.rotation_y += turn_dir * rotation_speed * time.dt

            move_vec = v_dir * self.forward
            if brake or (forward and backward):
                move_vec = Vec3(0, 0, 0)
            
            if self.mode == 'SEMI-AUTO' and move_vec.length() > 0:
                if self.check_collision(move_vec, env):
                    move_vec = Vec3(0,0,0)
                    self.collision_alert = True
                else:
                    self.collision_alert = False
                    
        elif self.mode == 'AUTO':
            # AI Navigation Logic
            if not self.returning_to_base:
                # Find nearest undetected target
                closest_target = None
                min_dist = 9999
                for t in env.targets:
                    if not t.detected:
                        d = (t.position - self.position).length()
                        if d < min_dist:
                            min_dist = d
                            closest_target = t
                if closest_target:
                    self.target_pos = closest_target.position
                    self.status = f"AUTO: SEARCHING FOR {closest_target.target_type.upper()}"
                else:
                    self.target_pos = None
                    self.status = "AUTO: AREA CLEAR - IDLE"

            if self.target_pos:
                diff = self.target_pos - self.position
                if diff.length() > 1.0:
                    move_vec = diff.normalized()
                    # Obstacle Avoidance logic
                    if self.check_collision(move_vec, env):
                        self.collision_alert = True
                        # Try alternate paths (left/right)
                        alt_left = self.left * 0.5 + move_vec * 0.5
                        alt_right = self.right * 0.5 + move_vec * 0.5
                        if not self.check_collision(alt_left, env):
                            move_vec = alt_left
                        elif not self.check_collision(alt_right, env):
                            move_vec = alt_right
                        else:
                            move_vec = -self.forward # Back up if stuck
                    else:
                        self.collision_alert = False
                elif self.returning_to_base and diff.length() < 2.0:
                    self.status = "AT BASE - MISSION HALTED"
                    self.returning_to_base = False
                    self.mode = 'MANUAL'

        # Apply final movement
        if move_vec.length() > 0:
            speed = ROVER_SPEED * time.dt
            if held_keys['shift']:
                speed *= 1.5
            zone = env.get_zone_at(self.position)
            if zone == 'flood': speed *= 0.3
            elif zone == 'landslide': speed *= 0.5
            
            self.position += move_vec * speed
            if move_vec.length() > 0.01:
                self.look_at(self.position + move_vec)
            
            # Drain battery
            drain = BATTERY_DRAIN_BASE
            if zone == 'fire': drain *= 6; self.status = "WARNING: FIRE ZONE - OVERHEATING!"
            self.battery -= drain

        # Keep rover always on the ground level
        self.y = 0.5

        # Random Comms Loss
        if not self.comms_lost and random.random() < 0.00005:
            self.comms_lost = True

    def check_collision(self, move_dir, env):
        # Raycast only for real obstacles (debris/blocks), not bodies
        origin = self.position + Vec3(0, 0.5, 0)
        # Use a slightly larger distance so the rover
        # keeps a safe margin and does not touch debris.
        hit_info = raycast(origin, move_dir, distance=4, ignore=(self,))
        if not hit_info.hit or hit_info.entity == self:
            return False
        # Treat collision only when we hit one of the registered obstacles
        return hit_info.entity in env.obstacles

# ==========================================
# ENVIRONMENT & TERRAIN
# ==========================================

class Environment:
    def __init__(self):
        # Ground
        self.ground = Entity(model='plane', scale=(200, 1, 200), texture='grass', color=color.dark_gray, collider='mesh')
        
        # ROAD NETWORK (Long, Turny, Intersections)
        self.roads = []
        # Main Road
        self.add_road(Vec3(0, 0.01, 0), Vec3(12, 0.1, 150)) # North-South
        self.add_road(Vec3(50, 0.01, 0), Vec3(150, 0.1, 12)) # East-West intersection
        self.add_road(Vec3(-40, 0.01, 40), Vec3(80, 0.1, 12), rotation=45) # Diagonal Road
        
        # Disaster Zones
        self.flood_zone = Entity(model='plane', scale=(40, 1, 40), color=color.blue, alpha=0.4, x=60, z=60, y=0.2)
        self.fire_zone = Entity(model='plane', scale=(30, 1, 30), color=color.orange, alpha=0.3, x=-60, z=-60, y=0.2)
        self.landslide_zone = Entity(model='plane', scale=(50, 1, 50), color=color.brown, alpha=0.5, x=-60, z=60, y=0.2)
        self.earthquake_zone = Entity(model='plane', scale=(50, 1, 50), color=color.gray, alpha=0.3, x=60, z=-60, y=0.2)
        
        # Obstacles (Impassable Debris)
        self.obstacles = []
        # Earthquake debris clusters
        for _ in range(40):
            pos = Vec3(random.uniform(40, 80), 0.5, random.uniform(-40, -80))
            obs = Entity(model='cube', position=pos, scale=(random.uniform(2,5), random.uniform(2,6), random.uniform(2,5)), 
                         texture='brick', color=color.gray, collider='box')
            self.obstacles.append(obs)
            
        # General obstacles
        for _ in range(50):
            pos = Vec3(random.uniform(-90, 90), 0.5, random.uniform(-90, 90))
            if pos.length() > 15:
                obs = Entity(model='cube', position=pos, scale=(random.uniform(1,4), random.uniform(1,5), random.uniform(1,4)), 
                             texture='brick', color=color.gray, collider='box')
                self.obstacles.append(obs)

        # Targets (Survivors, Injured, Dead)
        self.targets = []
        types = ['survivor', 'injured', 'dead']
        for _ in range(12):
            t_type = random.choice(types)
            pos = Vec3(random.uniform(-80, 80), 0, random.uniform(-80, 80))
            while self.get_zone_at(pos) == 'fire' and t_type != 'dead':
                pos = Vec3(random.uniform(-80, 80), 0, random.uniform(-80, 80))
            target = RescueTarget(target_type=t_type, position=pos)
            self.targets.append(target)

    def add_road(self, pos, scale, rotation=0):
        road = Entity(model='cube', position=pos, scale=scale, rotation_y=rotation, texture='noise', color=color.black, y=0.01)
        self.roads.append(road)

    def get_zone_at(self, pos):
        if (pos - self.flood_zone.position).length() < 20: return 'flood'
        if (pos - self.fire_zone.position).length() < 15: return 'fire'
        if (pos - self.landslide_zone.position).length() < 25: return 'landslide'
        if (pos - self.earthquake_zone.position).length() < 25: return 'earthquake'
        return None

# ==========================================
# MAIN SIMULATION
# ==========================================

env = Environment()
rover = TankRover(position=(0, 0.5, 0))

# Camera setup
camera.position = Vec3(0, 30, -40)
camera.rotation_x = 35

# UI Elements
mode_ui = Text(text='MODE: MANUAL', position=(-0.85, 0.45), origin=(-0.5, 0.5), color=color.lime, background=True)
battery_ui = Text(text='BATTERY: 100%', position=(-0.85, 0.40), origin=(-0.5, 0.5), color=color.lime, background=True)
status_ui = Text(text='STATUS: IDLE', position=(-0.85, 0.35), origin=(-0.5, 0.5), color=color.lime, background=True)
stats_ui = Text(text='FOUND: S:0 I:0 D:0', position=(-0.85, 0.30), origin=(-0.5, 0.5), color=color.lime, background=True)

def update():
    # Camera Follow
    camera.position = lerp(camera.position, rover.position + Vec3(0, 30, -35), time.dt * 3)
    
    # Rover logic
    rover.update_logic(env)
    
    # Sensor Detection logic: detect body when rover is within ~1 meter
    for t in env.targets:
        if not t.detected:
            dist = (rover.position - t.position).length()
            if dist < BODY_NEAR_DISTANCE:
                # Line of Sight check inside the near distance
                hit = raycast(rover.position + Vec3(0, 1, 0),
                              (t.position - rover.position).normalized(),
                              distance=BODY_NEAR_DISTANCE + 0.5,
                              ignore=(rover,))
                if hit.entity == t:
                    t.mark_detected()
                    rover.detected_count[t.target_type] += 1
                    pos_str = f"({int(t.x)}, {int(t.z)})"
                    rover.status = f"DETECTED {t.target_type.upper()} at {pos_str}!"

                    # COORDINATE POPUP WINDOW (2 SECONDS)
                    popup = WindowPanel(
                        title='BODY DETECTED',
                        content=(
                            Text(text=f'TYPE: {t.target_type.upper()}', color=color.black),
                            Text(text=f'LOCATION: {pos_str}', color=color.black),
                            Text(text='MISSION LOG UPDATED', color=color.black),
                        ),
                        position=(0, 0.2),
                        origin=(0, 0),
                        z=-1
                    )
                    invoke(destroy, popup, delay=2)
                    # Pause rover movement for 2 seconds so the operator
                    # can read the alert, then resume searching.
                    rover.pause_until = time.time() + 2
                    # In AUTO mode, clear current target so AI picks the next
                    # undetected body after the pause.
                    if rover.mode == 'AUTO' and not rover.returning_to_base:
                        rover.target_pos = None

    # UI Refresh
    mode_ui.text = f"MODE: {rover.mode}"
    battery_ui.text = f"BATTERY: {int(rover.battery)}%"
    status_ui.text = f"STATUS: {rover.status}"
    stats_ui.text = f"FOUND - S:{rover.detected_count['survivor']} I:{rover.detected_count['injured']} D:{rover.detected_count['dead']}"
    
    if rover.collision_alert:
        print_on_screen("COLLISION AVOIDANCE ACTIVE", position=(0, 0.35), scale=1.5, color=color.red, duration=0.05)

# Input mapping
def input(key):
    if key == '1': rover.mode = 'AUTO'; rover.returning_to_base = False
    elif key == '2': rover.mode = 'SEMI-AUTO'; rover.returning_to_base = False
    elif key == '3': rover.mode = 'MANUAL'; rover.returning_to_base = False
    elif key == 'c': rover.comms_lost = False; rover.status = "SIGNAL RESTORED"
    elif key == 'o':
        # Add dynamic obstacle
        new_obs = Entity(model='cube', position=rover.position + rover.forward * 6, scale=(3,3,3), 
               texture='brick', color=color.gray, collider='box')
        env.obstacles.append(new_obs)
    elif key == 'r':
        # Remove nearest obstacle
        if env.obstacles:
            closest = None
            min_dist = 10
            for obs in env.obstacles:
                dist = (rover.position - obs.position).length()
                if dist < min_dist:
                    min_dist = dist
                    closest = obs
            if closest:
                env.obstacles.remove(closest)
                destroy(closest)
                rover.status = "OBSTACLE REMOVED"
    elif key == 'q' or key == 'escape':
        application.quit()

# Environment effects
Sky()
DirectionalLight(y=2, z=3, x=1, rotation=(45, 45, 45))

# Tutorial text
Text(text="1:AUTO | 2:SEMI | 3:MANUAL | W/S: Forward-Back | A/D or ARROWS: Turn | Shift: Boost | Space: Brake | O: Add Obs | R: Rem Obs | C: Signal Fix", 
     position=(-0.5, -0.45), scale=0.8, color=color.white)

app.run()
