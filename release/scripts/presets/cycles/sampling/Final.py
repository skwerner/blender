import bpy
cycles = bpy.context.scene.cycles

# Path Trace
cycles.samples = 512
cycles.preview_samples = 128
