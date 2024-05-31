import numpy as np
import pandas as pd

from src.Extras.LineManager import *
from src.Extras.HelperFunctions import *
from src.Extras.Skybox import *
from src.Extras.Circle import *

from panda3d.core import AmbientLight , DirectionalLight , Point3 , MouseButton
from panda3d.core import Vec3 , KeyboardButton , TextureStage , TransparencyAttrib
from panda3d.core import LightAttrib , NodePath , CardMaker , NodePath , TextNode
from panda3d.core import AntialiasAttrib, loadPrcFileData

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from direct.gui.OnscreenImage import OnscreenImage



class MyApp(ShowBase):

    def __init__(self , data_path):
        ShowBase.__init__(self)
        self.anti_antialiasing(is_on=True)

        self.data = pd.read_csv(data_path)
        self.current_frame = 0

        self.n_frames = len(self.data)
        print(self.n_frames)

        row_0 = self.data.loc[self.current_frame]
        self.n_debris = len(row_0) - 3
        
        self.current_target = row_0['target_index']

        self.setup_scene()
        self.taskMgr.add(self.check_keys, "check_keys_task")
        self.accept("space" , self.on_space_pressed)
        self.game_is_paused = False
        self.accept("a" , self.on_a_pressed)
        
        self.taskMgr.doMethodLater(1/30, self.renderer, 'renderer')

        


    def setup_scene(self):
        
        setup_skybox(self.render , self.loader)
        self.setup_camera()
        self.setup_nodes()
        self.setup_lights()
        self.setup_hud()


    def renderer(self, task):
        
        if not self.game_is_paused:
            # self.otv.update(dt=DT)
            # self.target.update(dt=DT)

            
            
            rotate_object(self.earth , [0.05 , 0 , 0])
            # rotate_object(self.cloud , [0.05 , 0 , 0])
            self.env_visual_update()
            
            
                

        return Task.cont
        

    

    
        

    def env_visual_update(self):


        self.update_otv_trail()
        self.update_hud()


        line_pos = []
        for pos in self.otv_trail_nodes:
            if pos != (0,0,0):
                line_pos.append(pos)

        
        self.line_manager.update_line('trail_line', line_pos, color=(1, 0, 0, 1))

        
        # Frames updates
        current_row = self.data.loc[self.current_frame]
        
        self.current_frame += 1
        if self.current_frame == self.n_frames-1:
            self.current_frame = 0

            for debris_node in self.debris_nodes:
                debris_node.show()

            

        # otv
        otv_pos_str = current_row['otv'].strip("[]").split()
        otv_pos = np.array([float(num) for num in otv_pos_str])
        self.otv_node.setPos(otv_pos[0] , otv_pos[1] , otv_pos[2])

        # otv rotation
        next_row = self.data.loc[self.current_frame+1]
        otv_next_pos_str = next_row['otv'].strip("[]").split()
        otv_next_pos = np.array([float(num) for num in otv_next_pos_str])
        otv_dir = otv_next_pos - otv_pos
        otv_dir = otv_dir / np.linalg.norm(otv_dir)
        self.otv_node.setH(np.degrees(np.arctan2(otv_dir[1] , otv_dir[0])))
        self.otv_node.setP(90 + np.degrees(np.arcsin(otv_dir[2])))
        self.otv_node.setR(0)


        # debris
        for i in range(1 , self.n_debris):
            debris_i_pos_str = current_row[f'debris{i}'].strip("[]").split()
            debris_i_pos = np.array([float(num) for num in debris_i_pos_str])
            self.debris_nodes[i].setPos(debris_i_pos[0] , debris_i_pos[1] , debris_i_pos[2])

            # debris rotation
            debris_next_pos_str = next_row[f'debris{i}'].strip("[]").split()
            debris_next_pos = np.array([float(num) for num in debris_next_pos_str])
            debris_dir = debris_next_pos - debris_i_pos
            debris_dir = debris_dir / np.linalg.norm(debris_dir)
            self.debris_nodes[i].setH(np.degrees(np.arctan2(debris_dir[1] , debris_dir[0])))
            self.debris_nodes[i].setP(90 + np.degrees(np.arcsin(debris_dir[2])))
            self.debris_nodes[i].setR(0)
            



        current_fuel = current_row['fuel']
        self.fuel_label.setText(f"Fuel: {round(current_fuel/10,1)}%")

        if self.current_target != current_row['target_index']:
            print('switching target')
            print(self.current_target)
            print(current_row['target_index'])
            self.make_debris_trajectory(debris_id=current_row['target_index'])
            self.debris_nodes[self.current_target+1].hide()

        self.current_target = current_row['target_index']
        self.target_label.setText(f"Target: {self.current_target}")
            
        

    def setup_nodes(self):
        self.make_earth()

        self.otv_node = self.make_sphere(size=0.005 , otv=True)
        self.otv_node.reparentTo(self.render)

        self.debris_nodes = []
        for _ in range(self.n_debris):
            node = self.make_sphere(size=0.005 , sat=True)
            node.reparentTo(self.render)
            self.debris_nodes.append(node)




        self.otv_trail_nodes = []
        self.otv_trail_counter = 10
        self.make_otv_trail()

        self.line_manager = LineManager(self.render)
        self.line_manager.make_line('trail_line', [(0, 0, 0), (0, 0, 0)], color=(1, 0, 0, 1))
        

        self.setup_planes_visualisation()


        # make a circle
        # circle_points = make_circle(radius=1 , n_points=100 , center=(0,0,0) , rotation=(0,0,0))
        # self.line_manager.make_line('circle' , circle_points , color=(0,1,0,1) , thickness=2.0)

        self.make_debris_trajectory(debris_id=self.current_target)



    def make_object(self , elements):
        init_pos = [1,1,1]
        node = self.make_sphere(size=0.03 , low_poly=True)
        node.setPos(init_pos[0] , init_pos[1] , init_pos[2])
        node.reparentTo(self.render)

        
        return node
    


    def make_otv_trail(self):
        init_pos = (0,0,0) # At first, all the nodes are hidden inside the earth
        n = 100

        for _ in range(n):
            node = init_pos
            self.otv_trail_nodes.append(node)


    def update_otv_trail(self):
        # update every frame
        self.otv_trail_counter -= 1

        if self.otv_trail_counter <= 0:
            self.otv_trail_counter = 4
        else:
            return
        
        self.otv_trail_nodes[0] = self.otv_node.getPos()
        self.otv_trail_nodes = self.otv_trail_nodes[1:] + [self.otv_trail_nodes[0]]

    def make_debris_trajectory(self , debris_id):
        all_points = []
        for i in range(len(self.data)):
            pos = self.data.iloc[i][f'debris{debris_id+1}']
            pos = pos.strip('[]').split()
            pos = tuple([float(num) for num in pos])
            all_points.append(pos)

        self.line_manager.update_line(f'debris_trajectory' , all_points , color=(0,1,1,1) , thickness=0.5)

        
    def make_earth(self):
        self.earth = self.make_sphere(size=0.7)
        self.earth.setPos(0, 0, 0)
        self.earth.setHpr(0, 90, 0)

        # Load textures
        base_texture = self.loader.loadTexture("src/Assets/Textures/earth_albedo.jpg")
        gloss_texture = self.loader.loadTexture("src/Assets/Textures/earth_specular.jpg")
        glow_texture = self.loader.loadTexture("src/Assets/Textures/earth_emission.png")

        # Base Texture Stage
        base_ts = TextureStage('base')
        base_ts.setMode(TextureStage.MModulate)
        self.earth.setTexture(base_ts, base_texture)

        # Gloss Texture Stage
        gloss_ts = TextureStage('gloss')
        gloss_ts.setMode(TextureStage.MGloss)
        self.earth.setTexture(gloss_ts, gloss_texture)

        # Glow Texture Stage
        glow_ts = TextureStage('glow')
        glow_ts.setMode(TextureStage.MAdd)
        self.earth.setTexture(glow_ts, glow_texture)

        self.earth.reparentTo(self.render)




    def setup_lights(self):
        # Ambient light
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor((0.1, 0.1, 0.1, 1))
        ambient_light_node = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_light_node)

        # Directional light
        directional_light = self.render.attachNewNode("directional_light")
        d_light = DirectionalLight("d_light")
        intensity = 4
        d_light.setColor((intensity, intensity, intensity, 1))
        d_light_np = directional_light.attachNewNode(d_light)
        d_light_np.setHpr(45, -15, 0)
        self.render.setLight(d_light_np)


    def setup_camera(self):
        self.rotation_speed = 50.0
        self.elevation_speed = -50.0

        self.distance_to_origin = 10.0
        self.distance_speed = 0.1
        self.min_dist = 4
        self.max_dist = 16

        self.angle_around_origin = 0.0
        self.elevation_angle = 0.0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        self.camera.setPos(0, -10, 0)
        self.camera.lookAt(0, 0, 0)

        self.camLens.setNear(0.1)
        self.camLens.setFar(100.0)
        
        self.disableMouse() # Enable mouse control for the camera
        self.accept("mouse1", self.mouse_click)

        self.taskMgr.add(self.update_camera_task, "update_camera_task")

    def setup_planes_visualisation(self):

        # Axis planes
        s = (4,4)
        c = (0 , 0.51 , 0.71 , 0.3)
        self.visualisation_plane_1 = self.create_plane(size=s , position=(0,0,0) , rotation=(90,0,0) , color=c)
        self.visualisation_plane_2 = self.create_plane(size=s , position=(0,0,0) , rotation=(0,90,0) , color=c)
        self.visualisation_plane_3 = self.create_plane(size=s , position=(0,0,0) , rotation=(0,0,90) , color=c)

        self.visualisation_plane_1.hide()
        self.visualisation_plane_2.hide()
        self.visualisation_plane_3.hide()

        self.visualisation_plane_1.setAttrib(LightAttrib.makeAllOff())
        self.visualisation_plane_2.setAttrib(LightAttrib.makeAllOff())
        self.visualisation_plane_3.setAttrib(LightAttrib.makeAllOff())

        self.visualisation_plane_is_on = False

        

    def toggle_plane_visualisation(self):
        self.visualisation_plane_is_on = not self.visualisation_plane_is_on

        if self.visualisation_plane_is_on:
            self.visualisation_plane_1.show()
            self.visualisation_plane_2.show()
            self.visualisation_plane_3.show()
        else:
            self.visualisation_plane_1.hide()
            self.visualisation_plane_2.hide()
            self.visualisation_plane_3.hide()
    

    def create_plane(self, size, position, rotation, color=(1, 1, 1, 1)):
        card_maker = CardMaker('plane')
        w, h = size
        card_maker.setFrame(-w / 2, w / 2, -h / 2, h / 2)  # Set the size of the plane
        plane_np = NodePath(card_maker.generate())
        plane_np.reparentTo(self.render)
        plane_np.setPos(position)  # Set position
        plane_np.setHpr(rotation)  # Set rotation

        # Set color and alpha
        r, g, b, a = color  # Unpack the color tuple
        plane_np.setColor(r, g, b, a)  # Apply color and alpha
        plane_np.setTwoSided(True)
        plane_np.setTransparency(TransparencyAttrib.MAlpha)

        return plane_np
        


    def setup_hud(self):
        y_st = 0.9
        y_sp = 0.1
        x_po = -1.3
        self.label_1 = self.add_text_label(text="label 1" , pos=(x_po , y_st))

        self.pause_label = self.add_text_label(text="II" , pos=(0 , y_st))
        self.pause_label.hide()

        self.fuel_label = self.add_text_label(text="Fuel: #" , pos=(x_po , y_st - y_sp))
        self.target_label = self.add_text_label(text="Target: #" , pos=(x_po , y_st - 2*y_sp))
        

    def update_hud(self):
        self.label_1.setText(f"{self.current_frame}/{self.n_frames}")
    


    
    def mouse_click(self):
        # Check if the left mouse button is down
        if self.mouseWatcherNode.isButtonDown(MouseButton.one()):
            # Enable mouse motion task
            self.last_mouse_x = self.mouseWatcherNode.getMouseX()
            self.last_mouse_y = self.mouseWatcherNode.getMouseY()
            self.taskMgr.add(self.update_camera_task, "update_camera_task")

    def update_camera_task(self, task):
        # Check if the left mouse button is still down
        if self.mouseWatcherNode.isButtonDown(MouseButton.one()):
            # Get the mouse position
            current_mouse_x = self.mouseWatcherNode.getMouseX()
            current_mouse_y = self.mouseWatcherNode.getMouseY()

            # Check if the mouse has moved horizontally
            if current_mouse_x != self.last_mouse_x:
                # Adjust the camera rotation based on the mouse horizontal movement
                self.angle_around_origin -= (current_mouse_x - self.last_mouse_x) * self.rotation_speed

            # Check if the mouse has moved vertically
            if current_mouse_y != self.last_mouse_y:
                # Adjust the camera elevation based on the mouse vertical movement
                self.elevation_angle += (current_mouse_y - self.last_mouse_y) * self.elevation_speed
                self.elevation_angle = max(-90, min(90, self.elevation_angle))  # Clamp the elevation angle

            self.update_camera_position()

            self.last_mouse_x = current_mouse_x
            self.last_mouse_y = current_mouse_y

            return task.cont
        else:
            # Disable the mouse motion task when the left button is released
            self.taskMgr.remove("update_camera_task")
            return task.done
        

    def update_camera_position(self):
        radian_angle = np.radians(self.angle_around_origin)
        radian_elevation = np.radians(self.elevation_angle)
        x_pos = self.distance_to_origin * np.sin(radian_angle) * np.cos(radian_elevation)
        y_pos = -self.distance_to_origin * np.cos(radian_angle) * np.cos(radian_elevation)
        z_pos = self.distance_to_origin * np.sin(radian_elevation)

        self.camera.setPos(Vec3(x_pos, y_pos, z_pos))
        self.camera.lookAt(Point3(0, 0, 0))
        

    def check_keys(self, task):
        # Check if the key is down
        if self.mouseWatcherNode.is_button_down(KeyboardButton.up()):
            self.move_forward()
        if self.mouseWatcherNode.is_button_down(KeyboardButton.down()):
            self.move_backward()
        
        return task.cont

    def move_forward(self):
        if self.distance_to_origin > self.min_dist:
            self.distance_to_origin -= self.distance_speed
            self.update_camera_position()


    def move_backward(self):
        if self.distance_to_origin < self.max_dist:
            self.distance_to_origin += self.distance_speed
            self.update_camera_position()
        

    def make_sphere(self , size=1 , low_poly=False , otv=False , sat=False):
        path = "src/Assets/Models/sphere5.obj"
        if low_poly:
            path = "src/Assets/Models/low_poly_sphere.obj"
        if otv:
            path = "src/Assets/Models/otv.obj"
        if sat:
            path = "src/Assets/Models/sat.obj"
        sphere = self.loader.loadModel(path)
        sphere.setScale(size)
        return sphere
    

    def add_text_label(self , text="PlaceHolder" , pos=(-1 , 1) , scale=0.06 , alignment_mode=TextNode.ALeft):
        custom_font = self.loader.loadFont('src/Assets/Textures/SF-Pro.ttf')
        text_label = OnscreenText(text=text,
                                    pos=pos, # Position on the screen
                                    scale=scale, # Text scale
                                    fg=(1, 1, 1, 1), # Text color (R, G, B, A)
                                    bg=(0, 0, 0, 0.5), # Background color (R, G, B, A)
                                    align=alignment_mode, # Text alignment
                                    font=custom_font,
                                    mayChange=True) # Allow text to change dynamically
        return text_label
        
    def add_image(self, image_path, pos=(0, 0), scale=1, parent=None):
        if parent is None:
            parent = self.render2d  # Use self.render2d if no parent is specified.
        img = OnscreenImage(image=image_path, pos=pos, scale=scale, parent=parent)
        img.setTransparency(TransparencyAttrib.MAlpha)


    def on_a_pressed(self):
        self.toggle_plane_visualisation()

    def on_z_pressed(self):
        if self.visualisation_plane_4.isHidden():
            self.visualisation_plane_4.show()
        else:
            self.visualisation_plane_4.hide()

    def on_space_pressed(self):
        self.game_is_paused = not self.game_is_paused

        if self.game_is_paused:
            self.pause_label.show()
        else:
            self.pause_label.hide()

    def anti_antialiasing(self , is_on):
        if is_on:
            loadPrcFileData('', 'multisamples 4')  # Enable MSAA
            self.render.setAntialias(AntialiasAttrib.MAuto)
    


# if __name__ == "__main__":
#     app = MyApp(data_path="/Users/pierre/Documents/GitHub/Poliastro_Validation/data.csv")
#     app.run()