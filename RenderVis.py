from src.simulator.RenderEngine import MyApp

if __name__ == '__main__':
    data_path = 'src/vis_data/data_1.csv'
    app = MyApp(data_path)
    app.run()