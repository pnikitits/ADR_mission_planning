from src.simulator.RenderEngine import MyApp

if __name__ == '__main__':
    data_path = 'src/vis_data/data_1.csv'
    app = MyApp(data_path)
    app.run()



# from direct.showbase.ShowBase import ShowBase
# from panda3d.core import GraphicsEngine , GraphicsOutput , GraphicsPipe , WindowProperties

# class ShaderVersionChecker(ShowBase):
#     def __init__(self):
#         ShowBase.__init__(self)
#         gsg = self.win.getGsg()
#         shader_version = gsg.getDriverShaderVersionMajor()
#         print(f"Supported GLSL Shader Version: {shader_version}")
#         print(f'Graphics Renderer: {gsg.getDriverRenderer()}')
#         print(f'Graphics Vendor: {gsg.getDriverVendor()}')
#         print(f'Graphics Version: {gsg.getDriverVersion()}')

#         self.userExit()

# if __name__ == '__main__':
#     app = ShaderVersionChecker()
#     app.run()