"""Microelectronics scenes (viz README section 7).  From renderer/, render a still or a clip, e.g.:

    manim -ql -s examples/viz/micro_demo.py PNDemo        # one frame
    manim -ql    examples/viz/micro_demo.py MosfetDemo    # a short clip

Scenes: PNDemo, PNBandsDemo, MosfetDemo, PipelineDemo, BranchDemo, CacheDemo, HierarchyDemo, BusDemo, DmaDemo, SpiDemo, I2cDemo,
UartDemo, UsbDemo, CanDemo, DiffDemo.
"""
from manimate.ctx import Ctx
from manimate.viz.core import VizScene
from manimate.viz.microelectronics import (play_branch_prediction, play_cache_access, play_cpu_pipeline, play_memory_system,
                                           play_mosfet_channel, play_pn_junction, play_serial_bus)


def _scene(name: str, duration: float, fn, **kw):
    def script(self, ctx: Ctx) -> None:
        fn(ctx, **kw)

    return type(name, (VizScene,), {"duration": duration, "script": script, "__module__": __name__})


PNDemo = _scene("PNDemo", 12.0, play_pn_junction)
PNBandsDemo = _scene("PNBandsDemo", 10.0, play_pn_junction, view="bands")
MosfetDemo = _scene("MosfetDemo", 12.0, play_mosfet_channel)
PipelineDemo = _scene("PipelineDemo", 12.0, play_cpu_pipeline, program="load_use", forwarding=True)
BranchDemo = _scene("BranchDemo", 14.0, play_branch_prediction)
CacheDemo = _scene("CacheDemo", 14.0, play_cache_access)
HierarchyDemo = _scene("HierarchyDemo", 14.0, play_memory_system)
BusDemo = _scene("BusDemo", 10.0, play_memory_system, view="bus")
DmaDemo = _scene("DmaDemo", 10.0, play_memory_system, view="dma")
SpiDemo = _scene("SpiDemo", 10.0, play_serial_bus, variant="spi", data=[0xA5], mode=1)
I2cDemo = _scene("I2cDemo", 10.0, play_serial_bus, variant="i2c")
UartDemo = _scene("UartDemo", 8.0, play_serial_bus, variant="uart", mode=1)
UsbDemo = _scene("UsbDemo", 8.0, play_serial_bus, variant="usb", data=[0xFF, 0x0F])
CanDemo = _scene("CanDemo", 14.0, play_serial_bus, variant="can")
DiffDemo = _scene("DiffDemo", 8.0, play_serial_bus, variant="diff")
