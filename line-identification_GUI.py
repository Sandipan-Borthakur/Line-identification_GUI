import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, QTabWidget, QTableWidget,
                             QTableWidgetItem, QSplitter,
                             QCheckBox, QHeaderView, QPushButton)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector
from matplotlib.figure import Figure
import numpy as np
from astropy.io import fits
from scipy import signal
import os.path

def create_image_from_lines(lines,ncol):
    max_order = int(np.max(lines["order"]))
    img = np.zeros((max_order + 1, ncol))
    for line in lines:
        if line["order"] < 0:
            continue
        if line["xlast"] < 0 or line["xfirst"] > ncol:
            continue
        first = int(max(line["xfirst"], 0))
        last = int(min(line["xlast"], ncol))
        img[int(line["order"]), first:last] = line["height"] * signal.windows.gaussian(last - first, line["width"])
    return img

def find_offset(offset_info_path,lines,thar_master):
    if os.path.exists(offset_info_path):
        offset = np.loadtxt(offset_info_path)
        offset = [int(k) for k in offset]
    else:
        img = create_image_from_lines(lines, ncol=2088)
        correlation = signal.correlate2d(thar_master, img, mode="same")
        offset_order, offset_x = np.unravel_index(np.argmax(correlation), correlation.shape)

        offset_order = offset_order - img.shape[0] // 2 + 1
        offset_x = offset_x - img.shape[1] // 2 + 1
        offset = [int(offset_order), int(offset_x)]
        np.savetxt(offset_info_path, np.c_[int(offset_order), int(offset_x)], delimiter=' ', fmt=['%d', '%d'])
    return offset

def apply_alignment_offset(lines, offset, select=None):
    lines["xfirst"][select] += offset[1]
    lines["xlast"][select] += offset[1]
    lines["posm"][select] += offset[1]
    lines["order"][select] += offset[0]
    return lines

class PlotCanvas(FigureCanvas):

    def __init__(self, thar_master, lines, index=0, parent=None, width=5, height=4, dpi=100):
        self.thar_master = thar_master
        self.lines = lines
        self.orders = self.lines['order']
        self.positions = self.lines['posm']
        self.xfirsts = self.lines['xfirst']
        self.xlasts = self.lines['xlast']
        self.wavelengths = lines['wlc']
        self.pixels = np.arange(0, self.thar_master.shape[1])
        self.hidden_lines = set()  # Set to store hidden lines (track indices)

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(PlotCanvas, self).__init__(self.fig)

        # Plot based on index
        self.plot(index)

    def toggle_line(self, index, line_idx, checked):
        """Toggle the visibility of a specific line based on checkbox state."""
        if checked:
            self.hidden_lines.discard(line_idx)  # Remove from hidden lines if checked
        else:
            self.hidden_lines.add(line_idx)  # Add to hidden lines if unchecked
        self.plot(index)  # Re-plot to reflect changes

    def on_button_click(self):
        print(self.tempxmin,self.tempxmax)

    def onselect(self,xmin, xmax, line_idx):
        self.tempxmin,self.tempxmax = xmin,xmax
        # print(xmin,xmax,line_idx)

    def plot(self, index):
        self.button = QPushButton('Add line', self)
        self.button.clicked.connect(self.on_button_click)
        layout = QVBoxLayout()
        layout.addWidget(self.button)

        self.axes.clear()
        self.axes.plot(np.arange(0, self.thar_master.shape[1]), self.thar_master[index])

        if index in self.orders:
            ind = np.where(self.orders == index)[0]
            croppositions = self.positions[ind]
            cropxfirsts = self.xfirsts[ind]
            cropxlasts = self.xlasts[ind]
            cropwavelengths = self.wavelengths[ind]
            croporders = self.orders[ind]

            for j in range(len(ind)):
                if j in self.hidden_lines:  # Skip if the line is hidden
                    continue

                indlinepos = np.argmax(self.thar_master[index][cropxfirsts[j]:cropxlasts[j] + 1])
                linepos = self.thar_master[index][cropxfirsts[j]:cropxlasts[j] + 1][indlinepos]
                maxpixel = self.pixels[cropxfirsts[j] + indlinepos]
                self.axes.plot([maxpixel, maxpixel], [1.1 * linepos, 1.5 * linepos], 'r', lw=2)
                self.axes.text(maxpixel, 1.6 * linepos, '{:.4f}'.format(cropwavelengths[j]), rotation='vertical')

        self.span = SpanSelector(
            self.axes,
            lambda xmin,xmax:self.onselect(xmin,xmax,line_idx=index),
            "horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="tab:green"),
            interactive=True,
            drag_from_anywhere=True
        )

        self.axes.set_title(f'ThAr Master Order {index + 1}')
        self.axes.set_xlabel('Pixels')
        self.axes.set_ylabel('Intensity')
        self.axes.set_ylim((-400, max(self.thar_master[index]) + 400))
        self.draw()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Wavelength Calibration Line Identification")

        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)

        thar_master_path = '/export/borthaku/Codes/toes-pyreduce-pipeline/TOES-reduced/toes.thar_master.fits'
        hdu = fits.open(thar_master_path)
        thar_master = hdu[0].data

        linelist_path = '/export/borthaku/Codes/toes-pyreduce-pipeline/TOES-reduced/toes.linelist.npz'
        data = np.load(linelist_path, allow_pickle=True)
        lines = data['cs_lines']

        offset_info_path = 'offset_info.txt'

        offset = find_offset(offset_info_path, lines, thar_master)
        lines = apply_alignment_offset(lines, offset)

        for i in range(thar_master.shape[0]):
            tab = QWidget()
            layout = QVBoxLayout(tab)

            canvas = PlotCanvas(thar_master, lines, index=i, parent=self, width=5, height=4)
            toolbar = NavigationToolbar(canvas, self)

            linelist_table = QTableWidget()
            self.populate_linelist_table(linelist_table, lines, i, canvas)

            splitter = QSplitter()
            splitter.addWidget(canvas)
            splitter.addWidget(linelist_table)

            splitter.setStretchFactor(0, 2)
            splitter.setStretchFactor(1, 1)

            layout.addWidget(splitter)
            layout.addWidget(toolbar)

            tab_widget.addTab(tab, f"Order {i + 1}")

    def populate_linelist_table(self, table, lines, index, canvas):
        """Populate the QTableWidget with linelist data and add checkboxes to toggle line visibility."""
        order_lines = lines[lines['order'] == index]

        # Set number of rows and columns in the table
        table.setRowCount(len(order_lines))
        table.setColumnCount(6)  # Add one more column for the checkbox
        table.setHorizontalHeaderLabels(['Select','Order', 'Wavelength', 'Position', 'xFirst', 'xLast'])

        for i, line in enumerate(order_lines):
            # Add a checkbox to toggle line visibility
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Checked by default
            checkbox.stateChanged.connect(
                lambda state, line_idx=i: self.toggle_line(canvas, index, line_idx, state == 2))
            table.setCellWidget(i, 0, checkbox)  # Add the checkbox to the 0th column

            table.setItem(i, 1, QTableWidgetItem(f'{line["order"]:.0f}'))
            table.setItem(i, 2, QTableWidgetItem(f'{line["wlc"]:.4f}'))
            table.setItem(i, 3, QTableWidgetItem(f'{line["posm"]:.2f}'))
            table.setItem(i, 4, QTableWidgetItem(f'{line["xfirst"]:.0f}'))
            table.setItem(i, 5, QTableWidgetItem(f'{line["xlast"]:.0f}'))

        table.resizeColumnsToContents()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def toggle_line(self, canvas, index, line_idx, checked):
        """Call the canvas method to toggle the visibility of the line."""
        canvas.toggle_line(index, line_idx, checked)


def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()