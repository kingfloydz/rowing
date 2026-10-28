"""Raise the rigger ends 70 mm without moving the central hull attachment.

CAD approximation: preserve x/y and incline the outer struts in z. This is
visual geometry; the boat's existing aggregate mass/inertia are retained.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import trimesh
from hydrodynamics.models import ROOT


def main():
    mesh=trimesh.load_mesh(ROOT/'boat_meshes/rigger.stl')
    mesh.vertices[:,2] += .070*np.clip((np.abs(mesh.vertices[:,1])-.20)/.59,0,1)
    mesh.export(ROOT/'boat_meshes/rigger_raised.stl')


if __name__=='__main__':main()
