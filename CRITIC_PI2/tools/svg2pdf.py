import svglib
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
NAME = "reward"
drawing = svg2rlg(f"{NAME}.svg")
renderPDF.drawToFile(drawing, f"{NAME}.pdf")
renderPM.drawToFile(drawing, f"{NAME}.png", fmt="PNG")
renderPM.drawToFile(drawing, f"{NAME}.jpg", fmt="JPG")