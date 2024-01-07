from typing import List
from vispy import scene
from vispy.app import use_app, Application
from vispy.io import read_mesh, load_data_file
from vispy.scene.visuals import Mesh, GridLines, Line
from vispy.scene import transforms
from vispy.visuals.filters import ShadingFilter, WireframeFilter


class WireframeRenderer:
    def __init__(self, wireframe_width=1, shininess=100, backend="pyglet"):
        self.app = Application(backend_name=backend)
        self.app.create()
        self.canvas = scene.SceneCanvas(
            keys="interactive", bgcolor="white", app=self.app, size=(1280, 720)
        )
        self.view = self.canvas.central_widget.add_view()

        self.view.camera = "arcball"
        self.view.camera.depth_value = 1e3

        self.meshes = []
        self.meshes_data = []
        self.mesh_pos = []
        self.mesh_faces = []

        self.wireframe_filter = WireframeFilter(
            width=wireframe_width, wireframe_only=False, faces_only=False
        )
        self.shading_filter = ShadingFilter(shininess=shininess, shading=None)

        self.attach_headlight(self.view)

        self.init_axis()

        self.canvas.events.key_press.connect(self.on_key_press)

        # self.lines = [
        #     Line(
        #         pos=[[0, 0, 0], [1, 0, 0]],
        #         width=10,
        #         color="red",
        #         parent=self.view.scene,
        #     )
        # ]
    def on_key_press(self, event):
        if event.text == 'a':
            if len(self.mesh_pos) == 0:
                return
            self.mesh_pos[0] = (self.mesh_pos[0] + 1) % len(self.meshes_data[0])
            self.update_mesh(0, self.meshes_data[0][self.mesh_pos[0]], self.mesh_faces[0])
            mid_point = self.meshes_data[0][self.mesh_pos[0]].mean(axis=0)
            self.view.camera.center = mid_point
            pass
        elif event.text == 'b':
            pass

    def init_axis(self):
        scene.visuals.XYZAxis(parent=self.view.scene, width=10)

    def attach_headlight(self, view):
        light_dir = (0, 1, 0, 0)
        self.shading_filter.light_dir = light_dir[:3]
        initial_light_dir = view.camera.transform.imap(light_dir)

        @view.scene.transform.changed.connect
        def on_transform_change(event):
            transform = view.camera.transform
            self.shading_filter.light_dir = transform.map(initial_light_dir)[:3]

    def show(self):
        self.canvas.show()

    def run(self):
        self.app.run()

    def add_mesh(self, v, f, animate_data, color=(0.5, 0.7, 0.5, 1)):
        mesh = Mesh(v, f, color=color)
        self.view.add(mesh)
        mesh.attach(self.wireframe_filter)
        mesh.attach(self.shading_filter)
        self.meshes.append(mesh)
        self.mesh_faces.append(f)
        self.meshes_data.append(animate_data)
        self.mesh_pos.append(0)
        # return the index of the mesh
        return len(self.meshes) - 1

    def update_mesh(self, index, vert, face):
        mesh = self.meshes[index]
        mesh.set_data(vertices=vert, faces=face)
        mesh.update()
