"""Static, purely visual water surface for the Viser policy viewer."""
import numpy as np


def add_water_surface(server, height):
    """Add a world-horizontal 2 km square; no simulation geometry or callbacks.

    Current environments share the same water height and zero scene spacing.
    Two triangles suffice even for a 40 s traversal; no per-step GPU readback.
    The viewer translates /fixed_bodies when camera tracking recenters the
    scene, so the rendered water stays at its correct world-relative height.
    """
    surface = server.scene.add_mesh_simple(
        '/fixed_bodies/rowing_water/surface',
        vertices=np.array([[-1000,-1000,0],[1000,-1000,0],
                           [1000,1000,0],[-1000,1000,0]],dtype=np.float32),
        faces=np.array([[0,1,2],[0,2,3]],dtype=np.uint32),
        position=(0.,0.,float(height)),color=(55,145,185),opacity=.30,
        side='double',cast_shadow=False,receive_shadow=False)
    with server.gui.add_folder('Water surface'):
        show = server.gui.add_checkbox('Show water',initial_value=True)
        opacity = server.gui.add_slider('Opacity',min=0.,max=1.,step=.05,initial_value=.30)

        @show.on_update
        def _(_event):
            surface.visible = show.value

        @opacity.on_update
        def _(_event):
            surface.opacity = opacity.value

    return surface
