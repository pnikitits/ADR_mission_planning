from direct.showbase.ShowBase import ShowBase
from panda3d.core import LineSegs, NodePath



class LineManager:
    def __init__(self, render):
        self.render = render
        self.lines = {}

    def make_line(self, name, points, color=(1, 0, 0, 1), thickness=2.0):
        # Check if line already exists
        if name in self.lines:
            print(f"Line with name '{name}' already exists. Updating instead.")
            self.update_line(name, points, color, thickness)
            return

        # Create LineSegs object
        lines = LineSegs()
        lines.setThickness(thickness)
        lines.setColor(*color)

        # Add points to LineSegs
        for point in points:
            lines.moveTo(point) if point == points[0] else lines.drawTo(point)

        # Create NodePath
        line_geom_node = lines.create(False)
        node_path = NodePath(line_geom_node)
        node_path.reparentTo(self.render)

        # Store the line data
        self.lines[name] = (lines, node_path)

    def update_line(self, name, points, color=(1, 0, 0, 1), thickness=2.0):
        if name not in self.lines:
            # print(f"No line with name '{name}' found. Creating a new one.")
            self.make_line(name, points, color, thickness)
            return
        
        # Retrieve the existing LineSegs object; NodePath is not needed here
        lines, _ = self.lines[name]
        
        # Reset and update LineSegs with new properties and points
        lines.reset()
        lines.setThickness(thickness)
        lines.setColor(*color)
        for point in points:
            lines.moveTo(point) if point == points[0] else lines.drawTo(point)
        
        # Remove the old geom node from the scene graph
        self.lines[name][1].removeNode()
        
        # Create a new geom node and reparent it to the render
        line_geom_node = lines.create(False)
        node_path = NodePath(line_geom_node)
        node_path.reparentTo(self.render)

        # Update the stored NodePath for this line
        self.lines[name] = (lines, node_path)

    