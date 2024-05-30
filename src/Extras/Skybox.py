from panda3d.core import TextureStage



def setup_skybox(render , loader):
        size = 20
        distance = 20

        texture_list = []
        for i in range(1,7):
            texture_list.append(loader.loadTexture(f"src/Assets/Skybox/Skybox_{i}.jpg"))
        
        path = "src/Assets/Models/plane.obj"
        plane_1 = loader.loadModel(path)
        plane_2 = loader.loadModel(path)
        plane_3 = loader.loadModel(path)
        plane_4 = loader.loadModel(path)
        plane_5 = loader.loadModel(path)
        plane_6 = loader.loadModel(path)

        base_ts = TextureStage('base_ts')
        base_ts.setMode(TextureStage.MReplace)

        plane_1.setTexture(base_ts , texture_list[0])
        plane_1.setScale(size)
        plane_2.setTexture(base_ts , texture_list[0])
        plane_2.setScale(size)
        plane_3.setTexture(base_ts , texture_list[0])
        plane_3.setScale(size)
        plane_4.setTexture(base_ts , texture_list[0])
        plane_4.setScale(size)
        plane_5.setTexture(base_ts , texture_list[0])
        plane_5.setScale(size)
        plane_6.setTexture(base_ts , texture_list[0])
        plane_6.setScale(size)

        plane_1.setPos(0, 0, distance)
        plane_1.setHpr(0 , -90 , 0)

        plane_2.setPos(0, 0, -distance)
        plane_2.setHpr(0 , 90 , 0)

        plane_3.setPos(distance, 0, 0)
        plane_3.setHpr(90 , 0 , 0)

        plane_4.setPos(-distance, 0, 0)
        plane_4.setHpr(-90 , 0 , 0)

        plane_5.setPos(0, distance, 0)
        plane_5.setHpr(-180 , 0 , 0)

        plane_6.setPos(0, -distance, 0)
        plane_6.setHpr(0 , 0 , 90)

        plane_1.reparentTo(render)
        plane_2.reparentTo(render)
        plane_3.reparentTo(render)
        plane_4.reparentTo(render)
        plane_5.reparentTo(render)
        plane_6.reparentTo(render)