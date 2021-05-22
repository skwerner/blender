import bpy
cycles = bpy.context.scene.cycles

# Path Trace
cycles.samples = 128
cycles.preview_samples = 32
