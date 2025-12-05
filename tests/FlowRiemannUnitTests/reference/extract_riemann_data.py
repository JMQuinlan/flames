"""
VisIt script to extract 1D slice from 2D AMReX data
Run with: visit -nowin -cli -s extract_riemann_data.py
"""

import sys

# Configuration
case_name = 'Toro1a'
visit_file = '../../../bin/tests/FlowRiemannUnitTests/output_Toro1a/celloutput.visit'
output_csv = f'{case_name}_numerical_data.csv'
y_slice = 0.0  # Extract at y=0

# Open the database
OpenDatabase(visit_file, 0)

# Get the last timestep
SetTimeSliderState(TimeSliderGetNStates() - 1)

# Define the slice operator
AddOperator("Slice", 1)
SliceAtts = SliceAttributes()
SliceAtts.originType = SliceAtts.Point
SliceAtts.originPoint = (0, y_slice, 0)
SliceAtts.normal = (0, 1, 0)  # Slice perpendicular to y-axis
SliceAtts.project2d = 1
SetOperatorOptions(SliceAtts, 1)

# Extract velocityx
DeleteAllPlots()
AddPlot("Curve", "operators/Lineout/velocityx", 1, 1)
DrawPlots()

# Create lineout along x-axis
Lineout((-1, y_slice), (1, y_slice))

# Export velocityx
SetActiveWindow(2)
ExportDBAtts = ExportDBAttributes()
ExportDBAtts.db_type = "Curve2D"
ExportDBAtts.filename = "velocityx_data"
ExportDBAtts.dirname = "."
ExportDatabase(ExportDBAtts)

# Extract pressure
SetActiveWindow(1)
DeleteAllPlots()
AddPlot("Curve", "operators/Lineout/pressure", 1, 1)
DrawPlots()
Lineout((-1, y_slice), (1, y_slice))

SetActiveWindow(2)
ExportDBAtts.filename = "pressure_data"
ExportDatabase(ExportDBAtts)

# Extract density
SetActiveWindow(1)
DeleteAllPlots()
AddPlot("Curve", "operators/Lineout/density", 1, 1)
DrawPlots()
Lineout((-1, y_slice), (1, y_slice))

SetActiveWindow(2)
ExportDBAtts.filename = "density_data"
ExportDatabase(ExportDBAtts)

print("Data extraction complete!")
print(f"Files created: velocityx_data.curve, pressure_data.curve, density_data.curve")

sys.exit()
